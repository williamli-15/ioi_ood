import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib

matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


class CausalAbstraction:
    def __init__(self, cache, layers=[0, 5, 12, 23],
                 hook_points=["attn.hook_q", "attn.hook_attn_scores", "mlp.hook_post", "hook_resid_pre"]):
        self.cache = cache
        self.hook_points = hook_points
        self.layers = layers

    def mutual_information(self, x, y, bins=100):
        x = x.detach().cpu().numpy().flatten()
        y = y.detach().cpu().numpy().flatten()
        hist_2d, _, _ = np.histogram2d(x, y, bins=bins, density=True)
        p_xy = hist_2d / hist_2d.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        p_xy = p_xy + 1e-10
        p_x_p_y = np.outer(p_x, p_y) + 1e-10
        mi = np.sum(p_xy * np.log(p_xy / p_x_p_y))
        return mi if not np.isnan(mi) else 0.0

    def build_dag(self, scheme="id"):
        G = nx.DiGraph()
        nodes = []
        for layer in self.layers:
            for hook in self.hook_points:
                node = f"layer{layer}_{hook}_{scheme}"
                nodes.append(node)
                activation = self.cache.get(f"blocks.{layer}.{hook}", torch.tensor(0.0))
                G.add_node(node, activation=activation)

        for i in range(len(nodes) - 1):
            act1 = G.nodes[nodes[i]]["activation"].flatten()
            act2 = G.nodes[nodes[i + 1]]["activation"].flatten()
            weight = self.mutual_information(act1, act2) if len(act1) == len(act2) else 1.0
            G.add_edge(nodes[i], nodes[i + 1], weight=max(0.1, weight))
            if i % len(self.hook_points) == len(self.hook_points) - 1:
                if i + len(self.hook_points) < len(nodes):
                    G.add_edge(nodes[i], nodes[i + len(self.hook_points)], weight=0.5)
        return G

    def js_divergence(self, p, q):
        p = torch.softmax(p, dim=-1) + 1e-10
        q = torch.softmax(q, dim=-1) + 1e-10
        m = 0.5 * (p + q)
        return 0.5 * (torch.sum(p * torch.log(p / m)) + torch.sum(q * torch.log(q / m))).item()

    def compute_mediation(self, dag, faithfulness=None, scheme="id"):
        mediation_effects = {}
        for layer in self.layers:
            try:
                attn = dag.nodes[f"layer{layer}_attn.hook_q_{scheme}"]["activation"]
                mlp = dag.nodes[f"layer{layer}_mlp.hook_post_{scheme}"]["activation"]
                mi = self.mutual_information(attn, mlp)
                mediation = mi * (faithfulness or 1.0)
                mediation_effects[f"layer{layer}"] = mediation if np.isfinite(mediation) else 0.0
            except KeyError as e:
                print(f"警告: 节点 {e} 缺失，设中介效应为 0")
                mediation_effects[f"layer{layer}"] = 0.0
        return mediation_effects

    def compute_intervention_score(self, id_dag, ood_dag, scheme="ood"):
        scores = {}
        for node in id_dag.nodes:
            ood_node = node.replace("_id", f"_{scheme}")
            if ood_node in ood_dag.nodes:
                id_act = id_dag.nodes[node]["activation"].flatten()
                ood_act = ood_dag.nodes[ood_node]["activation"].flatten()
                score = self.js_divergence(id_act, ood_act)
                scores[node] = score if np.isfinite(score) else 0.0
        return scores

    def visualize_dag(self, dag, filename="dag.png", scheme="id"):
        pos = nx.spring_layout(dag)
        plt.figure(figsize=(12, 8))
        nx.draw(dag, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=8, font_weight='bold')
        nx.draw_networkx_edge_labels(dag, pos)
        plt.title(f"因果抽象 DAG（重要层激活，方案：{scheme})")
        plt.savefig(filename)
        plt.close()

    def visualize_intervention_scores(self, scores, filename="intervention_scores.png", scheme="ood"):
        plt.figure(figsize=(12, 6))
        nodes = list(scores.keys())
        values = list(scores.values())
        sns.heatmap([values], xticklabels=nodes, cmap="viridis", annot=True)
        plt.title(f"重要层 OOD 干预评分（方案：{scheme})")
        plt.savefig(filename)
        plt.close()

    def validate_mediation(self, id_mediation, ood_mediation, filename="mediation_comparison.png", scheme="ood"):
        plt.figure(figsize=(10, 6))
        id_values = [id_mediation[f"layer{l}"] for l in self.layers]
        ood_values = [ood_mediation[f"layer{l}"] for l in self.layers]
        plt.plot(self.layers, id_values, label="ID Mediation", marker='o')
        plt.plot(self.layers, ood_values, label=f"OOD Mediation ({scheme})", marker='x')
        plt.legend()
        plt.title(f"重要层中介效应比较 (ID vs OOD {scheme})")
        plt.xlabel("Layer")
        plt.ylabel("Mediation Effect")
        plt.savefig(filename)
        plt.close()
        t_stat, p_val = stats.ttest_ind(id_values, ood_values)
        return p_val

    def compare_multi_schemes(self, id_mediation, ood_mediations, schemes, filename="multi_scheme_mediation.png"):
        plt.figure(figsize=(12, 8))
        plt.plot(self.layers, [id_mediation[f"layer{l}"] for l in self.layers], label="ID Mediation", marker='o')
        for scheme, ood_mediation in zip(schemes, ood_mediations):
            plt.plot(self.layers, [ood_mediation[f"layer{l}"] for l in self.layers], label=f"OOD Mediation ({scheme})",
                     marker='x')
        plt.legend()
        plt.title("多组 OOD 方案中介效应比较")
        plt.xlabel("Layer")
        plt.ylabel("Mediation Effect")
        plt.savefig(filename)
        plt.close()