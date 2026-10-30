"""
Command-Line Interface (CLI) for Cross-Organization Differentially Private NIDS.
Supports automated training, evaluation, ablation benchmarking, and privacy audits.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np

# Ensure terminal printing succeeds across all Windows codepages
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dp_crossorg_nids.data.network_traffic_generator import generate_cross_organization_dataset
from dp_crossorg_nids.data.preprocessor import setup_federated_dataloaders
from dp_crossorg_nids.models.classifier import ResidualFlowClassifier
from dp_crossorg_nids.models.autoencoder import DeepAnomalyAutoencoder
from dp_crossorg_nids.federated.client import CrossOrgClient
from dp_crossorg_nids.federated.server import FederatedServer
from dp_crossorg_nids.attacks.membership_inference import MembershipInferenceAuditor
from dp_crossorg_nids.evaluation.metrics import compute_nids_metrics
from dp_crossorg_nids.evaluation.benchmark import run_benchmark_suite


def main():
    parser = argparse.ArgumentParser(
        description="Differentially Private Deep Learning for Cross-Organization Network Anomaly Detection"
    )
    parser.add_argument("--mode", type=str, choices=["train", "benchmark", "audit"], default="train",
                        help="Execution mode: 'train' (single FL run), 'benchmark' (privacy-utility ablation), 'audit' (MIA vulnerability test)")
    parser.add_argument("--model-type", type=str, choices=["classifier", "autoencoder"], default="classifier",
                        help="Deep learning model architecture")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs per client per round")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--noise-multiplier", type=float, default=1.0, help="DP Gaussian noise scale sigma (0 for non-private)")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="L2 gradient clipping threshold C")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target DP delta failure parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save weights and logs")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print(" Differentially Private Cross-Organization Network Anomaly Detection")
    print("="*70)
    print(f" Mode:            {args.mode.upper()}")
    print(f" Model Type:      {args.model_type.upper()}")
    print(f" FL Rounds:       {args.rounds}")
    print(f" DP Noise Scale:  sigma = {args.noise_multiplier}")
    print(f" Gradient Clip C: {args.clip_norm}")
    print(f" Target Delta:    {args.delta}")
    print("="*70 + "\n")

    if args.mode == "benchmark":
        print("Launching Privacy-Utility Benchmark Suite...")
        results = run_benchmark_suite(rounds=args.rounds, local_epochs=args.local_epochs, seed=args.seed)
        print("\n=== Benchmark Summary ===")
        print(results["summary_table"].to_string(index=False))

        summary_path = os.path.join(args.output_dir, "benchmark_summary.csv")
        results["summary_table"].to_csv(summary_path, index=False)
        print(f"\n[+] Benchmark exported to: {summary_path}")
        return

    # Train / Audit mode
    print("[1/4] Generating cross-organization network traffic...")
    data_dict = generate_cross_organization_dataset(seed=args.seed)
    for org, df in data_dict.items():
        print(f"  * {org}: {len(df)} flows | Anomaly rate: {df['is_anomaly'].mean():.2%}")

    print("\n[2/4] Setting up preprocessing and client partitions...")
    pipeline, client_loaders, global_test_loader = setup_federated_dataloaders(
        data_dict, batch_size=args.batch_size, seed=args.seed
    )
    input_dim = pipeline.feature_dim
    print(f"  * Total feature dimensions: {input_dim}")

    # Build model
    if args.model_type == "classifier":
        base_model = ResidualFlowClassifier(input_dim=input_dim, num_classes=6)
    else:
        base_model = DeepAnomalyAutoencoder(input_dim=input_dim, latent_dim=8)

    # Initialize clients
    clients = []
    for org_id, loaders in client_loaders.items():
        client = CrossOrgClient(
            org_id=org_id,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"],
            input_dim=input_dim,
            model_type=args.model_type,
            learning_rate=args.lr,
            clip_norm=args.clip_norm,
            noise_multiplier=args.noise_multiplier,
            delta=args.delta
        )
        clients.append(client)

    server = FederatedServer(
        global_model=base_model,
        global_test_loader=global_test_loader,
        clients=clients,
        model_type=args.model_type
    )

    print("\n[3/4] Starting Federated DP-FedAvg Training...")
    for r in range(1, args.rounds + 1):
        round_info = server.run_round(round_num=r, local_epochs=args.local_epochs)
        m = round_info["eval_metrics"]
        eps = round_info["epsilon_spent"]
        eps_str = f"eps = {eps:.2f}" if eps != float("inf") else "eps = inf (Non-Private)"
        print(f"  Round {r:02d}/{args.rounds:02d} | Test Loss: {m['loss']:.4f} | Acc/Score: {m['accuracy']:.2%} | Macro F1: {m['f1_macro']:.4f} | Privacy Spent: {eps_str} (delta={args.delta})")

    # Final Evaluation & Metrics
    print("\n[4/4] Conducting Final Evaluation & Privacy Audit...")
    final_eval = server.evaluate_global()

    if args.model_type == "classifier":
        nids_metrics = compute_nids_metrics(
            np.array(final_eval["ground_truth"]),
            np.array(final_eval["predictions"])
        )
        print("\n--- Final Intrusion Detection Performance ---")
        print(f"  Overall Accuracy:    {nids_metrics['accuracy']:.2%}")
        print(f"  Detection Rate (DR): {nids_metrics['detection_rate']:.2%}")
        print(f"  False Alarm Rate (FAR): {nids_metrics['false_alarm_rate']:.2%}")
        print(f"  Macro F1-Score:      {nids_metrics['f1_macro']:.4f}")

    # Privacy Audit
    org1_key = list(client_loaders.keys())[0]
    m_x, _, m_cls = pipeline.transform_df(client_loaders[org1_key]["raw_train"])
    nm_x, _, nm_cls = pipeline.transform_df(client_loaders[org1_key]["raw_test"])

    auditor = MembershipInferenceAuditor(server.global_model, model_type=args.model_type)
    audit_res = auditor.audit_privacy_leakage(m_x, m_cls, nm_x, nm_cls)

    print("\n--- Empirical Privacy Audit (Membership Inference Attack) ---")
    print(f"  Attacker ROC-AUC:    {audit_res['attack_auc']:.3f} (Ideal safe baseline: ~0.50)")
    print(f"  Attacker Accuracy:   {audit_res['attack_accuracy']:.2%}")
    print(f"  Privacy Risk Level:  {audit_res['risk_level']}")
    print(f"  Audit Verdict:       {audit_res['verdict']}")

    # Save trained global model
    model_save_path = os.path.join(args.output_dir, f"global_{args.model_type}_dp.pt")
    torch.save(server.global_model.state_dict(), model_save_path)
    print(f"\n[+] Saved global model weights to: {model_save_path}")

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    export_data = {
        "history": server.history,
        "privacy_audit": {
            "attack_auc": audit_res["attack_auc"],
            "risk_level": audit_res["risk_level"],
            "verdict": audit_res["verdict"]
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"[+] Saved execution telemetry to: {metrics_path}\n")


if __name__ == "__main__":
    main()
