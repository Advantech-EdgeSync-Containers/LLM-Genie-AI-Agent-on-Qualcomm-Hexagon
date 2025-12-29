## Windows Git Line Ending Setup

When using Windows, Git may automatically convert line endings from **LF (`\n`)** to **CRLF (`\r\n`)**, which can cause shell scripts (`.sh`), configuration files (`.env`, `.yml`), or other text files to break when transferred to Linux systems.  

To avoid this, you need to configure Git to preserve **LF line endings**.

---

### Set Git to preserve LF line endings

Run the following command in your terminal or PowerShell before cloning the repository:

```bash
git config --global core.autocrlf input
```

Please follow next steps as it is from main readme.