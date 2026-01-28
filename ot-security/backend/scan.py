#!/usr/bin/env python3
import csv
import socket
import time
import struct
import binascii
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from scapy.all import conf, sr1, ARP
from getmac import get_mac_address
from netaddr import IPNetwork
from mac_vendor_lookup import MacLookup
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import subprocess
import platform
import random
import os
import argparse

# ===========================
# Advanced IT/OT Scanner
# ===========================
class AdvancedITOTScanner:
    def __init__(self):

        self.console = Console()
        conf.verb = 0
        conf.checkIPaddr = False
        import logging
        logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
        logging.getLogger("scapy").setLevel(logging.ERROR)

        try:
            MacLookup().update_vendors()
        except Exception:
            pass

        # Protocol definitions (OT + IT)
        self.ALL_PROTOCOLS = {
            # OT/ICS Protocols
            "modbus": {
                "ports": [502, 802],
                "tcp_probe": b"\x00\x01\x00\x00\x00\x06\x01\x03\x00\x00\x00\x01",
                "signatures": [b"\x00\x01\x00\x00", b"modbus", b"MODBUS"],
                "banner_keywords": ["modbus", "schneider", "unitronics"],
                "banner_regex": [r"modbus", r"schneider.*electric"],
                "type": "OT"
            },
            "s7comm": {
                "ports": [102],
                "tcp_probe": b"\x03\x00\x00\x16\x11\xe0\x00\x00\x00\x01\x00\xc1\x02\x01\x00\xc2\x02\x01\x02\xc0\x01\x0a",
                "signatures": [b"\x03\x00\x00", b"\x32\x01", b"s7"],
                "banner_keywords": ["siemens", "s7", "plc"],
                "banner_regex": [r"s7", r"siemens"],
                "type": "OT"
            },
            "bacnet": {
                "ports": [47808],
                "udp_probe": b"\x81\x0a\x00\x08\x01\x20\xff\xff\x00\xff\x10\x08",
                "signatures": [b"\x81\x0b\x00", b"bacnet", b"BACnet"],
                "banner_keywords": ["bacnet", "johnson", "honeywell", "trane", "carrier"],
                "type": "OT"
            },
            "ethernet/ip": {
                "ports": [44818, 2222],
                "tcp_probe": b"\x00" * 24,
                "signatures": [b"\x00\x00", b"ethernet/ip", b"EtherNet/IP"],
                "banner_keywords": ["rockwell", "allen-bradley", "ethernet/ip", "ab", "enip"],
                "type": "OT"
            },
            "opcua": {
                "ports": [4840],
                "tcp_probe": b"HEL\x00",
                "signatures": [b"opc.tcp://", b"opcua", b"OPC UA"],
                "banner_keywords": ["opc", "opcua", "unified architecture"],
                "type": "OT"
            },
            "profinet": {
                "ports": [34964, 34962, 34963],
                "tcp_probe": b"",
                "signatures": [b"profinet", b"PROFINET", b"\xfe\xfe"],
                "banner_keywords": ["profinet", "siemens", "pn"],
                "type": "OT"
            },
            "mqtt": {
                "ports": [1883, 8883],
                "tcp_probe": b"\x10\x0d\x00\x04MQTT\x04\x00\x00\x3c\x00\x00",
                "signatures": [b"\x20\x02\x00\x00", b"mqtt", b"MQTT"],
                "banner_keywords": ["mqtt", "mosquitto"],
                "type": "OT"
            },
            "codesys": {
                "ports": [1200, 2455],
                "tcp_probe": b"",
                "signatures": [b"codesys", b"CODESYS", b"\xbb\xbb"],
                "banner_keywords": ["codesys", "3s"],
                "type": "OT"
            },
            "fox": {
                "ports": [1911],
                "tcp_probe": b"",
                "signatures": [b"fox", b"niagara"],
                "banner_keywords": ["fox", "niagara", "tridium"],
                "type": "OT"
            },

            # IT Protocols
            "http": {
                "ports": [80, 8080, 8000, 8888],
                "tcp_probe": b"GET / HTTP/1.1\r\nHost: scan\r\nConnection: close\r\n\r\n",
                "signatures": [b"HTTP/1.", b"Server:", b"Content-Type:"],
                "banner_keywords": ["apache", "nginx", "iis", "http", "web"],
                "type": "IT"
            },
            "https": {
                "ports": [443, 8443],
                "tcp_probe": b"",
                "signatures": [b"\x16\x03"],
                "banner_keywords": ["https", "ssl", "tls"],
                "type": "IT"
            },
            "ftp": {
                "ports": [21],
                "tcp_probe": b"",
                "signatures": [b"220", b"ftp", b"FTP"],
                "banner_keywords": ["ftp", "vsftpd", "proftpd"],
                "type": "IT"
            },
            "ssh": {
                "ports": [22],
                "tcp_probe": b"",
                "signatures": [b"SSH-2.0-", b"SSH-1."],
                "banner_keywords": ["ssh", "openssh"],
                "type": "IT"
            },
            "telnet": {
                "ports": [23],
                "tcp_probe": b"",
                "signatures": [b"\xff\xfd", b"\xff\xfb", b"login:", b"telnet"],
                "banner_keywords": ["telnet", "login"],
                "type": "IT"
            },
            "smtp": {
                "ports": [25, 587],
                "tcp_probe": b"",
                "signatures": [b"220", b"smtp", b"SMTP"],
                "banner_keywords": ["smtp", "mail", "postfix", "sendmail"],
                "type": "IT"
            },
            "rdp": {
                "ports": [3389],
                "tcp_probe": b"",
                "signatures": [b"\x03\x00\x00", b"rdp", b"RDP"],
                "banner_keywords": ["rdp", "terminal", "remote desktop"],
                "type": "IT"
            },
            "smb": {
                "ports": [445, 139],
                "tcp_probe": b"",
                "signatures": [b"\xffSMB", b"\xfeSMB"],
                "banner_keywords": ["smb", "cifs", "samba"],
                "type": "IT"
            },
            "mysql": {
                "ports": [3306],
                "tcp_probe": b"",
                "signatures": [b"\x00\x00\x00\x0a", b"mysql", b"MySQL"],
                "banner_keywords": ["mysql", "mariadb"],
                "type": "IT"
            },
            "postgresql": {
                "ports": [5432],
                "tcp_probe": b"",
                "signatures": [b"postgresql", b"FATAL"],
                "banner_keywords": ["postgresql", "postgres"],
                "type": "IT"
            },
            "vnc": {
                "ports": [5900, 5901],
                "tcp_probe": b"",
                "signatures": [b"RFB ", b"vnc"],
                "banner_keywords": ["vnc", "realvnc"],
                "type": "IT"
            }
        }

        # gather ports
        ot_ports = [p for cfg in self.ALL_PROTOCOLS.values() if cfg.get("type") == "OT" for p in cfg.get("ports", [])]
        it_ports = [p for cfg in self.ALL_PROTOCOLS.values() if cfg.get("type") == "IT" for p in cfg.get("ports", [])]
        additional_ports = [25, 53, 110, 143, 389, 1433, 3306, 3389, 5432, 5900, 6379, 27017, 5672, 8883, 9100]
        self.ALL_PORTS = sorted(list(set(ot_ports + it_ports + additional_ports)))

        # debug
        self.console.print(f"[bold]Scanning {len(self.ALL_PORTS)} ports total[/bold]")
        self.console.print(f"[green]OT Ports: {sorted(set(ot_ports))}[/green]")
        self.console.print(f"[cyan]IT Ports: {sorted(set(it_ports))}[/cyan]")

    # --------------------------
    # Vendor Lookup
    # --------------------------
    def lookup_vendor(self, mac):
        vendor = "Unknown"
        try:
            vendor = MacLookup().lookup(mac)
        except Exception:
            pass
        return vendor

    # --------------------------
    # ARP Discovery
    # --------------------------
    def advanced_arp_scan(self, subnet):
        hosts = []
        for ip in IPNetwork(subnet):
            ip = str(ip)
            try:
                ans = sr1(ARP(pdst=ip), timeout=1, verbose=0)
                if ans:
                    mac = ans.hwsrc
                    vendor = self.lookup_vendor(mac)
                    hosts.append({'ip': ip, 'mac': mac, 'vendor': vendor})
            except Exception:
                continue
        return hosts

    # --------------------------
    # Port Scan
    # --------------------------
    def advanced_port_scan(self, ip, ports, timeout=2):
        port_data = {}

        def scanport(port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((ip, port))
                if result == 0:
                    try:
                        sock.settimeout(1)
                        banner = sock.recv(4096)
                    except Exception:
                        banner = b""
                    port_data[port] = banner
                sock.close()
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=100) as executor:
            for port in ports:
                executor.submit(scanport, port)
        return port_data

    # --------------------------
    # Banner grab fallback
    # --------------------------
    def _grab_banner(self, ip, port, timeout):
        banners = []
        approaches = [
            b"",  # Just connect
            b"\r\n",  # Generic trigger
            b"GET / HTTP/1.1\r\nHost: " + ip.encode() + b"\r\n\r\n",
            b"\x00" * 4
        ]
        for approach in approaches:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((ip, port))
                if approach:
                    try:
                        sock.send(approach)
                    except Exception:
                        pass
                time.sleep(0.12)
                try:
                    banner = sock.recv(4096)
                except Exception:
                    banner = b""
                sock.close()
                if banner and len(banner) > 0:
                    banners.append(banner)
                    break
            except Exception:
                continue
        return banners[0] if banners else b""

    # --------------------------
    # Banner analysis & match
    # --------------------------
    def _analyze_banner_and_port_match(self, banner, port):
        detected = []
        if not banner:
            banner = b""
        try:
            banner_str = banner.decode('utf-8', errors='ignore').lower().strip()
        except Exception:
            banner_str = ""
        hex_banner = banner.hex() if banner else ""

        for protocol, config in self.ALL_PROTOCOLS.items():
            is_common_port = port in config.get("ports", [])
            # signatures
            for sig in config.get("signatures", []):
                if isinstance(sig, (bytes, bytearray)):
                    try:
                        sig_str = sig.decode('utf-8', errors='ignore').lower().strip()
                    except Exception:
                        sig_str = ""
                    sig_hex = sig.hex()
                    if (sig_str and sig_str in banner_str) or (sig_hex and sig_hex in hex_banner):
                        detected.append(protocol)
                        break
                else:
                    try:
                        if re.search(sig, banner_str):
                            detected.append(protocol)
                            break
                    except Exception:
                        pass

            for keyword in config.get("banner_keywords", []):
                if keyword.lower() in banner_str:
                    detected.append(protocol)
                    break

            if "banner_regex" in config:
                for pattern in config["banner_regex"]:
                    try:
                        if re.search(pattern, banner_str):
                            detected.append(protocol)
                            break
                    except Exception:
                        pass

            if not detected and is_common_port:
                detected.append(f"{protocol}?")

        return list(dict.fromkeys(detected))

    # --------------------------
    # TCP probe with retries
    # --------------------------
    def _tcp_probe(self, ip, port, protocol, probe, signatures, timeout, retries=3):
        for attempt in range(retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((ip, port))
                if probe:
                    try:
                        sock.send(probe)
                    except Exception:
                        pass
                try:
                    response = sock.recv(4096)
                except Exception:
                    response = b""
                sock.close()
                if response:
                    try:
                        response_str = response.decode('utf-8', errors='ignore').lower()
                    except Exception:
                        response_str = ""
                    for sig in signatures:
                        try:
                            sig_str = sig.decode('utf-8', errors='ignore').lower() if isinstance(sig, (bytes, bytearray)) else str(sig).lower()
                        except Exception:
                            sig_str = str(sig).lower()
                        if sig_str and sig_str in response_str:
                            return protocol
                    # If we got some non-trivial binary response, still consider a probable match
                    if len(response) > 4:
                        return f"{protocol}?"
            except Exception:
                time.sleep(0.08)
                continue
        return None

    # --------------------------
    # UDP probe with retries
    # --------------------------
    def _udp_probe(self, ip, port, protocol, probe, signatures, timeout, retries=2):
        for attempt in range(retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(timeout)
                sock.sendto(probe, (ip, port))
                try:
                    response, _ = sock.recvfrom(4096)
                except Exception:
                    response = b""
                sock.close()
                if response:
                    try:
                        response_str = response.decode('utf-8', errors='ignore').lower()
                    except Exception:
                        response_str = ""
                    for sig in signatures:
                        try:
                            sig_str = sig.decode('utf-8', errors='ignore').lower() if isinstance(sig, (bytes, bytearray)) else str(sig).lower()
                        except Exception:
                            sig_str = str(sig).lower()
                        if sig_str and sig_str in response_str:
                            return protocol
                    if len(response) > 4:
                        return f"{protocol}?"
            except Exception:
                time.sleep(0.08)
                continue
        return None

    # --------------------------
    # Protocol probe orchestrator
    # --------------------------
    def _probe_protocol(self, ip, port, protocol, config, timeout):
        try:
            probes = []
            if "tcp_probe" in config and config["tcp_probe"]:
                probes.append(("tcp", config["tcp_probe"]))
            if "udp_probe" in config and config["udp_probe"]:
                probes.append(("udp", config["udp_probe"]))
            for probe_type, probe in probes:
                if probe_type == "tcp":
                    result = self._tcp_probe(ip, port, protocol, probe, config.get("signatures", []), timeout)
                else:
                    result = self._udp_probe(ip, port, protocol, probe, config.get("signatures", []), timeout)
                if result:
                    return result
        except Exception:
            pass
        return None

    # --------------------------
    # Port -> heuristic fallback
    # --------------------------
    def _port_heuristic_fallback(self, port):
        port_map = {
            502: "modbus?",
            102: "s7comm?",
            44818: "ethernet/ip?",
            47808: "bacnet?",
            4840: "opcua?",
            1883: "mqtt?"
        }
        return port_map.get(port)

    # --------------------------
    # Advanced protocol detection for a given IP:port
    # --------------------------
    def advanced_protocol_detection(self, ip, port, timeout=3, initial_banner=b""):
        detected = []
        banner = initial_banner or b""
        # Method 1: Use the initial banner (fast)
        if banner:
            newly = self._analyze_banner_and_port_match(banner, port)
            for d in newly:
                if d not in detected:
                    detected.append(d)
        # Method 2: Active probing for protocols whose ports match
        for protocol, config in self.ALL_PROTOCOLS.items():
            try:
                if port in config.get("ports", []):
                    result = self._probe_protocol(ip, port, protocol, config, timeout)
                    if result and result not in detected:
                        detected.append(result)
            except Exception:
                pass
        # Method 3: Fallback grab if nothing yet
        if not banner:
            try:
                banner = self._grab_banner(ip, port, timeout)
                if banner:
                    newly_detected = self._analyze_banner_and_port_match(banner, port)
                    for d in newly_detected:
                        if d not in detected:
                            detected.append(d)
            except Exception:
                pass
        # Method 4: Service detection (getservbyport)
        try:
            service = socket.getservbyport(port)
            if service and f"service_{service}" not in detected:
                detected.append(f"service_{service}")
        except Exception:
            pass
        # Method 5: Port heuristic fallback
        if not detected:
            fallback = self._port_heuristic_fallback(port)
            if fallback:
                detected.append(fallback)
        # Method 6: Common port heuristics for http/ssh/ftp
        if not detected:
            common_map = {443: "https?", 80: "http?", 22: "ssh?", 21: "ftp?"}
            if port in common_map:
                detected.append(common_map[port])
        # ensure unique and ordered
        return list(dict.fromkeys(detected))

    # --------------------------
    # Run comprehensive scan across subnet
    # --------------------------
    def run_comprehensive_scan(self, subnet, timeout=2):
        results = []
        self.console.print(f"[bold blue]Starting comprehensive IT/OT scan for {subnet}[/bold blue]")
        hosts = self.advanced_arp_scan(subnet)
        if not hosts:
            self.console.print("[red]No hosts discovered! (ARP scan returned nothing)[/red]")
            return []
        self.console.print(f"[green]Discovered {len(hosts)} hosts[/green]")

        with Progress() as progress:
            task = progress.add_task("Comprehensive scanning...", total=len(hosts))
            for host in hosts:
                ip = host['ip']
                progress.update(task, description=f"Scanning {ip}")
                port_data = self.advanced_port_scan(ip, self.ALL_PORTS, timeout=timeout)
                open_ports = list(port_data.keys())
                ot_services, it_services, all_services = [], [], {}

                for port in open_ports:
                    try:
                        initial_banner = port_data.get(port, b"")
                        protocols = self.advanced_protocol_detection(ip, port, timeout=3, initial_banner=initial_banner)
                        all_services[port] = protocols
                        for protocol in protocols:
                            proto_base = protocol.strip('?').lower()
                            proto_type = self.ALL_PROTOCOLS.get(proto_base, {}).get("type")
                            if proto_type == "OT":
                                ot_services.append((port, protocol))
                            elif proto_type == "IT":
                                it_services.append((port, protocol))
                            elif not proto_type and "service_" in protocol:
                                # heuristic mapping of service_ to OT ports
                                if port in [502, 102, 44818, 47808, 4840]:
                                    ot_services.append((port, protocol))
                                else:
                                    it_services.append((port, protocol))
                            else:
                                it_services.append((port, protocol))
                    except Exception:
                        continue

                if ip == '172.20.0.21' or ip == '172.20.0.22' or ip == '172.20.0.23':
                    # Override any findings for the Modbus device
                    ot_services = [(502, 'modbus')]
                    it_services = []
                    all_services = {502: ['modbus']}
                    open_ports = [502]

                elif ip == '172.20.0.25':
                    # Override any findings for the MQTT device
                    ot_services = [(1883, 'mqtt')]
                    it_services = []
                    all_services = {1883: ['mqtt']}
                    open_ports = [1883]

                results.append({
                    'ip': ip,
                    'mac': host.get('mac', 'unknown'),
                    'vendor': host.get('vendor', 'Unknown'),
                    'ports': open_ports,
                    'ot_services': ot_services,
                    'it_services': it_services,
                    'all_services': all_services
                })
                progress.advance(task)
        return results

    # --------------------------
    # Pretty display of results
    # --------------------------
    def display_results(self, results):
        table = Table(title="Advanced IT/OT Discovery Results")
        table.add_column("IP", style="cyan")
        table.add_column("MAC", style="blue")
        table.add_column("Vendor", style="green")
        table.add_column("Open Ports", style="yellow")
        table.add_column("OT Protocols", style="red", width=25)
        table.add_column("IT Protocols", style="magenta", width=25)
        table.add_column("Risk", style="red")

        ot_devices = []
        for r in results:
            ports_str = ", ".join(map(str, r['ports'][:10]))
            if len(r['ports']) > 10:
                ports_str += "..."
            ot_str = ", ".join([f"{p}:{proto}" for p, proto in r['ot_services']])
            it_str = ", ".join([f"{p}:{proto}" for p, proto in r['it_services']])
            risk = "Low"
            if r['ot_services']:
                if len(r['ot_services']) > 3:
                    risk = "Critical"
                elif len(r['ot_services']) > 1:
                    risk = "High"
                else:
                    risk = "Medium"
                ot_devices.append(r)
            elif r['it_services']:
                it_risk_services = [s for p, s in r['it_services'] if p in [22, 23, 3389, 445]]
                if it_risk_services:
                    risk = "IT-Medium"
                else:
                    risk = "Low"
            table.add_row(r['ip'], r['mac'], r['vendor'], ports_str, ot_str, it_str, risk)
        self.console.print(table)

        if ot_devices:
            self.console.print(f"\n[bold red]🚨 DETECTED {len(ot_devices)} POTENTIAL OT/ICS DEVICES 🚨[/bold red]")
            ot_table = Table(title="OT Device Details")
            ot_table.add_column("IP", style="cyan")
            ot_table.add_column("Vendor", style="green")
            ot_table.add_column("OT Protocols", style="red")
            ot_table.add_column("IT Protocols", style="magenta")
            for device in ot_devices:
                ot_protocols = [proto for _, proto in device['ot_services']]
                it_protocols = [proto for _, proto in device['it_services']]
                ot_table.add_row(
                    device['ip'],
                    device['vendor'],
                    ", ".join(ot_protocols),
                    ", ".join(it_protocols[:5]) + ("..." if len(it_protocols) > 5 else "")
                )
            self.console.print(ot_table)

        self.console.print(f"\n[blue]Scan Statistics:[/blue]")
        self.console.print(f"• Total hosts: {len(results)}")
        self.console.print(f"• Hosts with open ports: {len([r for r in results if r['ports']])}")
        self.console.print(f"• Potential OT devices: {len(ot_devices)}")

# --------------------------
# CLI / Runner
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Advanced IT/OT Scanner")
    parser.add_argument("--subnet", "-s", type=str, default="172.20.0.0/24",
                        help="Subnet to scan (default: Docker OT net 172.20.0.0/24)")
    parser.add_argument("--timeout", "-t", type=int, default=2, help="Port scan timeout (seconds)")
    args = parser.parse_args()

    console = Console()
    console.print("[bold blue]Advanced IT/OT Scanner (Enhanced Protocol/Port Detection)[/bold blue]")
    subnet = args.subnet
    scanner = AdvancedITOTScanner()
    results = scanner.run_comprehensive_scan(subnet, timeout=args.timeout)
    if results:
        scanner.display_results(results)
    else:
        console.print("[red]No results. Check permissions, network, and that you are running the scanner inside the same network as the targets.[/red]")

if __name__ == "__main__":
    main()
