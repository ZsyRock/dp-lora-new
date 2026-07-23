# Executable DP-LoRA paper reconstruction

The upstream `main` branch is an incomplete prototype.  The
`hpc/paper-repro` branch adds a fail-closed, single-GPU simulation of the five
logical clients in Algorithm 1 of
“Differentially Private Low-Rank Adaptation of Large Language Model Using
Federated Learning” (arXiv:2312.17493).

The formal runner uses BERT-base and GPT-2 small sequentially on the English
MedDialog HealthcareMagic+iCliniq data.  It fixes the values explicitly stated
by the paper: `K=5`, `T=50`, `B=8`, `sigma=2`, `lr=5e-4`, `C=10`, and LoRA
rank `512`.  It updates both LoRA A and B, clips their aggregate batch-gradient
groups separately, adds Gaussian noise, and equally averages the five local
adapter states after every round.  It contains no SlaClip code path.

The paper omits model revisions, exact LoRA targets/alpha/dropout, optimizer,
sequence length, seed, and enough privacy-accounting constants to reproduce its
epsilon claims.  This branch records the chosen assumptions in every
`run_config.json`; it is an algorithm-level reconstruction, not a claim of
bit-for-bit reproduction.  Exact client batch losses and gradient statistics
are labelled `NON_DP_PRIVATE_DIAGNOSTIC`.

The executable is `paper_repro/train_federated.py`; immutable input staging is
implemented in `scripts/stage_paper_inputs.py`.  Cluster-specific launch files
are intentionally kept outside this Git worktree under
`$HOME/hpc/projects/dp-lora-paper/`.

# Upstream project description

The financial industry has experienced significant strides in Natural Language Processing (NLP) facilitated by Language Model (LM) technologies. However, the escalating concerns regarding data privacy present a formidable barrier to the ongoing enhancement of these models. A notable challenge involves potential adversarial attackers exploiting the weight of Language Models trained by individual banks, thereby jeopardizing user data confidentiality.

This project seeks to address the critical issue of data privacy in Language Model training within the financial sector by proposing and implementing a Privacy-Preserving Federated Learning Protocol.
The primary goal is to establish a collaborative framework that empowers multiple banks to collectively train a Language Model without the necessity to share or access each other's private and sensitive data. Departing from the conventional practice of sharing precise model weights, this innovative framework facilitates the exchange of "biased weights." This approach thwarts third-party attempts to infer training data, thereby safeguarding the confidentiality of sensitive information.

The core principle of this federated learning approach is to ensure that the Language Model remains robust, accurate, and reflective of the diverse financial data landscape. Simultaneously, it addresses the privacy concerns inherent to individual financial institutions. By mitigating data privacy risks, this project strives to foster an environment where advancements in NLP can continue to flourish in the financial sector, promoting collaborative innovation while upholding the highest standards of data security.

![image](https://github.com/Michonster/FinLLM-DP-Lora/assets/83566627/700b1274-2bec-41c8-a717-57d1b7036165)

Process and planning for this project is based on this paper: https://arxiv.org/abs/2312.17493
