# Cloud Services API Security - Frontend

This is a frontend application for the Cloud Services API Security project. It is built with Next.js 15 and Tailwind CSS.
## File Structure

```
frontend/
├── app/                     # Next.js App Router structure
│   ├── (default)/           # Main application routes
│   │   ├── dashboard/       # Dashboard page
│   │   ├── anyproxy/        # Data Collection page
│   │   ├── zsl/             # Zero-shot learning pages
│   │   │   └── deberta/     # DeBERTa model interface
│   │   ├── codebert/        # CodeBERT model interface
│   │   ├── rfc/             # Random Forest & Code generation
│   │   └── files/           # File browser interface
│   ├── api/                 # Backend API routes
│   └── layout.tsx           # Root layout
├── components/              # Reusable components
│   ├── ui/                  # Base UI components
│   ├── global/              # Global components (header, etc.)
│   ├── dashboard/           # Dashboard-specific components
│   ├── anyproxy/            # AnyProxy components
│   ├── deberta/             # DeBERTa components
│   ├── codebert/            # CodeBERT components
│   ├── rfc/                 # RFC components
│   ├── files/               # File browser components
│   └── landing/             # Landing page components
├── lib/                     # Utility functions and hooks
├── public/                  # Static assets
└── styles/                  # Global styles
```

## Getting Started

First, run the development server:

```bash
# Install dependencies
npm install

# Run the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.


