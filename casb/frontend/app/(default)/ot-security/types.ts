export interface ScanResult {
    ip: string;
    mac: string;
    vendor: string;
    open_ports: number[];
    ot_services: [number, string][];
    it_services: [number, string][];
    risk: string;
    port_count: number;
}
