import os
import csv
import dpkt
import socket
from datetime import datetime

# Folder containing .pcap files
PCAP_DIR = "/folder containing pcap files"
OUTPUT_CSV = "csvname.csv"

# Write CSV header if file doesn't exist
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
            "flow_duration", "total_fwd_bytes", "total_bwd_bytes", "total_pkts",
            "total_fwd_pkts", "total_bwd_pkts", "bytes_per_sec", "pkts_per_sec",
            "fwd_bwd_ratio", "label"
        ])

# Iterate through all .pcap files in the folder
for fname in os.listdir(PCAP_DIR):
    if not fname.endswith(".pcap"):
        continue

    pcap_path = os.path.join(PCAP_DIR, fname)
    print(f" Processing: {fname}")

    try:
        with open(pcap_path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            flows = {}

            for ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue

                    ip = eth.data
                    if not isinstance(ip.data, dpkt.udp.UDP):
                        continue  # QUIC uses UDP

                    udp = ip.data
                    src_ip = socket.inet_ntoa(ip.src)
                    dst_ip = socket.inet_ntoa(ip.dst)
                    src_port = udp.sport
                    dst_port = udp.dport
                    payload_len = len(udp.data)

                    key_fwd = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
                    key_bwd = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}"

                    if key_bwd in flows:
                        f = flows[key_bwd]
                        f["total_bwd_bytes"] += payload_len
                        f["total_bwd_pkts"] += 1
                        f["end_time"] = ts
                    else:
                        if key_fwd not in flows:
                            flows[key_fwd] = {
                                "src_ip": src_ip,
                                "dst_ip": dst_ip,
                                "src_port": src_port,
                                "dst_port": dst_port,
                                "protocol": "UDP(QUIC)",
                                "total_fwd_bytes": 0,
                                "total_bwd_bytes": 0,
                                "total_fwd_pkts": 0,
                                "total_bwd_pkts": 0,
                                "start_time": ts,
                                "end_time": ts,
                            }
                        f = flows[key_fwd]
                        f["total_fwd_bytes"] += payload_len
                        f["total_fwd_pkts"] += 1
                        f["end_time"] = ts

                except Exception:
                    continue  # skip malformed packets

            # Write results for this pcap file
            with open(OUTPUT_CSV, "a", newline="") as out_f:
                writer = csv.writer(out_f)
                for f in flows.values():
                    duration = max(f["end_time"] - f["start_time"], 1e-6)
                    total_pkts = f["total_fwd_pkts"] + f["total_bwd_pkts"]
                    total_bytes = f["total_fwd_bytes"] + f["total_bwd_bytes"]
                    bytes_per_sec = total_bytes / duration
                    pkts_per_sec = total_pkts / duration
                    fwd_bwd_ratio = (
                        f["total_fwd_bytes"] / f["total_bwd_bytes"]
                        if f["total_bwd_bytes"] > 0
                        else f["total_fwd_bytes"]
                    )

                    writer.writerow([
                        fname,
                        f["src_ip"],
                        f["dst_ip"],
                        f["src_port"],
                        f["dst_port"],
                        f["protocol"],
                        f"{duration:.6f}",
                        f["total_fwd_bytes"],
                        f["total_bwd_bytes"],
                        total_pkts,
                        f["total_fwd_pkts"],
                        f["total_bwd_pkts"],
                        f"{bytes_per_sec:.6f}",
                        f"{pkts_per_sec:.6f}",
                        f"{fwd_bwd_ratio:.6f}",
                        1  # label
                    ])

        print(f" Saved {len(flows)} flows from {fname}\n")

    except Exception as e:
        print(f" Error reading {fname}: {e}")

print(f" All PCAPs processed. Results stored in {OUTPUT_CSV}")
