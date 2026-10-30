# 📡 DrillGuard — Unsupervised Drilling Anomaly Detector

**Zero-label anomaly detection combining Isolation Forest + Deep Autoencoder Reconstruction on surface drilling signals.**

Part of the **Drilling ML Progression Series** on the **Equinor Volve Field (Well F-9A, North Sea)**:
1. 🔥 **[Axial Flare](https://axial-flare-copilot.streamlit.app/)** — Supervised Regression (*"How much ROP?"*)
2. 🛡️ **[DrillSense](https://drillsense-copilot.streamlit.app/)** — Supervised Classification (*"Which known dysfunction is happening?"*)
3. 🔮 **DrillCast** — Sequence Deep Learning / Bi-LSTM (*"What state comes next in the 30-ft window?"*)
4. 📡 **DrillGuard** — Unsupervised Learning (*"What if an abnormal event happens that was NEVER labeled?"*)

---

## 🧠 Unsupervised Architecture (Zero Labels Used in Training)

- **Hybrid Detector:** Deep Autoencoder (`15 → 10 → 4 → 10 → 15`) + Isolation Forest (`200` trees)
- **Validation Performance:** `0.997` ROC-AUC · `93.4%` F1 Score on post-hoc physical anomalies
- **Root-Cause Attribution:** Ranks which of the 15 surface & physics features deviates most from the normal drilling manifold

---

## 👤 Author

**Romer Villalobos**  
Senior Drilling & Completion Engineer | ML Engineer | Houston, TX  
[LinkedIn](https://www.linkedin.com/in/romer-villalobos-2b33a653)
