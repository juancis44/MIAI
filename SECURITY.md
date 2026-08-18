# Security Policy

## Supported Versions

MIAI is pre-1.0 software (`0.x.y`, no LTS or patch branches -- see
[docs/coding_standards.md](docs/coding_standards.md), "Versioning and
commits"). Every release so far has been a `0.x.0` bump against a single
`main` branch, so only the **latest published release** receives security
updates. There is no long-term support for older `0.x` releases; upgrading
to the latest release is the supported way to get a fix.

| Version         | Supported          |
| ---------------- | ------------------ |
| Latest `0.x.y`    | :white_check_mark: |
| Older `0.x.y`     | :x:                 |

This table will be revisited once MIAI reaches `1.0` and a stable-release
policy is defined.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's private vulnerability reporting for this repository:
go to the **Security** tab -> **Advisories** -> **Report a vulnerability**.
This opens a private conversation with the maintainer that isn't visible to
the public until a fix is ready.

What to expect:

- **Acknowledgement:** within 5 business days of your report.
- **Triage:** we'll confirm whether the report is accepted (a real,
  in-scope vulnerability) or declined (not reproducible, out of scope, or
  already known), and let you know which.
- **If accepted:** we'll work on a fix, coordinate a disclosure timeline
  with you, and credit you in the fix's release notes/changelog entry
  unless you prefer to stay anonymous.
- **If declined:** we'll explain why (e.g. not exploitable, intended
  behavior, or out of scope for this project).

Since MIAI is a research/tooling library (not a hosted service), most
realistic vulnerability classes are dependency-related (see
[`pip-audit`'s weekly scan](.github/workflows/security.yml)) or unsafe
deserialization/loading of untrusted model checkpoints or config files --
reports in those areas are especially welcome.
