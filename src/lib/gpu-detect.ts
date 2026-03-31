export interface GpuInfo {
  supported: boolean;
  adapterName: string;
  vendor: string;
  architecture: string;
  maxBufferSize: number;
  maxComputeWorkgroupSize: [number, number, number];
  maxComputeInvocationsPerWorkgroup: number;
  features: string[];
}

export async function detectGPU(): Promise<GpuInfo> {
  const unsupported: GpuInfo = {
    supported: false,
    adapterName: "",
    vendor: "",
    architecture: "",
    maxBufferSize: 0,
    maxComputeWorkgroupSize: [0, 0, 0],
    maxComputeInvocationsPerWorkgroup: 0,
    features: [],
  };

  if (typeof navigator === "undefined" || !navigator.gpu) {
    return unsupported;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return unsupported;

    const device = await adapter.requestDevice();
    const info = adapter.info;

    const features: string[] = [];
    adapter.features.forEach((f) => features.push(f));

    const result: GpuInfo = {
      supported: true,
      adapterName: info.device || info.description || (info.vendor ? `${info.vendor} ${info.architecture || "GPU"}` : "Unknown GPU"),
      vendor: info.vendor || "Unknown",
      architecture: info.architecture || "",
      maxBufferSize: device.limits.maxBufferSize,
      maxComputeWorkgroupSize: [
        device.limits.maxComputeWorkgroupSizeX,
        device.limits.maxComputeWorkgroupSizeY,
        device.limits.maxComputeWorkgroupSizeZ,
      ],
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
      features,
    };

    device.destroy();
    return result;
  } catch {
    return unsupported;
  }
}
