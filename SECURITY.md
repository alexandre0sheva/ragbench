# Security Policy

## Supported Versions

RAGBench is pre-1.0. Security fixes are applied to the latest version on the default branch.

## Reporting a Vulnerability

Please do not open a public issue for vulnerabilities involving credential exposure, prompt injection paths that disclose secrets, or unsafe handling of private documents.

Report security concerns through GitHub Security Advisories if enabled for the repository, or contact the maintainers through the repository owner profile.

## Secrets and Data

- Do not commit `.env` files or API keys.
- Use `.env.example` as the public template.
- Demo data must be fictional and safe to publish.
- Generated `results/` may contain model outputs and source text previews. Review them before sharing publicly.

## External Services

RAGBench can call OpenAI APIs when `OPENAI_API_KEY` is configured. Mock mode is available for local and CI runs without paid services:

```bash
ragbench compare --config configs/all.yaml --mock
```

