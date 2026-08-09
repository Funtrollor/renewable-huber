# Security policy

## Supported versions

`renewable-huber` remains pre-1.0 software, but published releases receive security fixes on the
latest patch of the current minor series. Development fixes also land on `main` on a best-effort
basis.

| Version | Supported |
| --- | --- |
| 0.6.1 (latest 0.6 patch) | Yes |
| latest `main` | Best effort |
| 0.5.x and earlier | No |

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/Funtrollor/renewable-huber/security/advisories/new)
and include:

- affected version or commit;
- impact and realistic attack scenario;
- minimal reproduction or proof of concept;
- suggested mitigation, if known.

The maintainer will acknowledge a complete report within 7 days, provide a preliminary assessment
within 14 days, and coordinate disclosure after a fix is available. These targets are best-effort
for a volunteer-maintained pre-1.0 project.

## Scope

Security reports should describe a confidentiality, integrity, or availability impact. Ordinary
numerical accuracy bugs, unsupported inputs, and performance regressions belong in the public issue
tracker unless disclosing them would create a security risk.
