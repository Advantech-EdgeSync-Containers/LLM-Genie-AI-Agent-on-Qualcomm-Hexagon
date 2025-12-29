# SCP Troubleshooting Guide  
This guide summarizes common issues and fixes related to failed SCP transfers, especially when uploading **large model files** (e.g., LLMs) to a Qualcomm device.

## Problem Summary
During SCP upload of a large file, the transfer starts normally but suddenly terminates with errors such as:

- `Connection reset`
- `client_loop: send disconnect: Connection reset`
- `scp.exe: Couldn't send packet: Broken pipe`

This prevents the full file from being copied to the target device.

## Common Causes & How to Fix Them

### 1. Network Timeout / Idle Disconnects
Office or enterprise networks often enforce aggressive timeouts.
- Increase `ServerAliveInterval` and `ServerAliveCountMax`  
  ```bash
  scp -o ServerAliveInterval=60 -o ServerAliveCountMax=10 <file> user@host:/path/
  ```
- Try another transfer tool like:
  ```bash
    rsync -avzP <file> user@host:/path/
  ```

### 2. Firewall / Proxy Restrictions on Large Transfers

Many corporate networks limit long-running SCP sessions or cap sustained upload bandwidth. 

- Try using a mobile hotspot or non-office network to rule out firewall restrictions.
- Try compressing the file first and then transfer

>Important: If you are on an office network, check with your internal IT team. Many companies enforce SCP/SFTP session timeout policies, bandwidth throttling, packet inspection, or file-size limitations that can interrupt large transfers.

### 3. Server-Side Limits on the Device

The Qualcomm device (or remote host) may have - Low SSH session buffer size, Limited storage & Running background processes causing drops

- Ensure enough free space on the target
- Restart SSH service on target device if possible
- Restart the device

### 4. Weak Wi-Fi or Unstable Connection

Large files over Wi-Fi frequently experience packet loss leading to disconnects.

- Prefer Ethernet for multi-GB model transfers.
- If using Wi-Fi, move closer to router or switch to 5 GHz (is applicable).

### 5. Windows OpenSSH Client Issues

The Windows built-in SCP client may fail intermittently with large files.
Use an alternative SCP/SFTP tool like:

- WinSCP
- MobaXterm
- WSL SCP