# 🩺 VisiHealth AI: Medical Visual Question Answering

**VisiHealth AI** is an advanced Medical Visual Question Answering (VQA) system designed to provide intelligent, text-based insights from medical imagery. By leveraging state-of-the-art vision and language models, VisiHealth allows users to ask natural language questions about medical scans (e.g., MRI, CT, X-Ray) and receive accurate, clinically relevant answers alongside Region of Interest (ROI) localization.

### 🌟 Key Features
- **Dual-Encoder Architecture**: Seamlessly fuses visual features from medical scans with deep semantic text embeddings using **BioLinkBERT**, a model pre-trained specifically on biomedical literature.
- **Multi-Token Cross-Attention**: Implements a sophisticated cross-attention modality fusion mechanism to ensure the model focuses on the exact regions of the image relevant to the specific clinical question being asked.
- **Multi-Task Learning**: Simultaneously trained for accurate Question Answering and semantic Region of Interest (ROI) localization to provide interpretable, honest diagnostic feedback.
- **Question-Aware Re-ranking**: Integrates a custom medical knowledge graph mapping system to intelligently re-rank and validate predicted answers based on the organ/condition context of the query.
- **High Performance**: Validated on the challenging **SLAKE dataset**, achieving strong baseline performance (~74.36% validation accuracy) utilizing techniques like Focal Loss for hard examples and `WeightedRandomSampler` for class balancing.
- **Full-Stack Application**: Includes a scalable PyTorch/FastAPI backend and a highly responsive, modern Next.js frontend for real-time inference and analysis.

### 🏗️ Tech Stack
- **Backend**: Python, PyTorch, FastAPI, Hugging Face Transformers
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **Models**: BioLinkBERT, ResNet/DenseNet Vision Encoder

### 🚀 Getting Started

#### Prerequisites
- Python 3.9+
- Node.js 18+

#### 1. Backend Setup
```bash
# Navigate to backend directory (or root if scripts are at root)
pip install -r requirements.txt

# Run the backend server
python scripts/demo.py  # Or the relevant FastAPI run command
```

#### 2. Frontend Setup
```bash
# Navigate to the frontend directory
cd visihealth-frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

### 🧠 Project Background
This project was developed as a Final Year Project (FYP). It bridges the gap between medical imaging and natural language processing to create an interactive, AI-driven diagnostic assistant.

---
*Note: Due to file size limits, trained model checkpoints (`.pth` files) and raw datasets (`data/`) are not included in this repository and must be downloaded/generated locally.*
