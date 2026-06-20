import { getUserConversions, updateConversionStatus, getConversionById } from "./db";
import { convertFile, isConversionSupported } from "./conversion-service";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STORAGE_DIR = path.resolve(__dirname, "..", "uploads");

/**
 * 后台转换任务处理器
 * 处理待转换的任务
 */

const POLL_INTERVAL = 2000; // 2 秒轮询一次（提高效率）
let isRunning = false;
const processingConversions = new Set<number>(); // 跟踪正在处理的任务

// 存储上传的文件内容，因为路由和 worker 之间不共享文件系统
const uploadedFiles: Map<number, Buffer> = new Map();

export function storeUploadedFile(conversionId: number, buffer: Buffer) {
  uploadedFiles.set(conversionId, buffer);
}

/**
 * 启动转换任务处理器
 */
export async function startConversionWorker() {
  if (isRunning) return;
  isRunning = true;

  console.log("[ConversionWorker] Started");

  const processLoop = async () => {
    try {
      await processPendingConversions();
    } catch (error) {
      console.error("[ConversionWorker] Error:", error);
    }

    if (isRunning) {
      setTimeout(processLoop, POLL_INTERVAL);
    }
  };

  processLoop();
}

/**
 * 停止转换任务处理器
 */
export function stopConversionWorker() {
  isRunning = false;
  console.log("[ConversionWorker] Stopped");
}

/**
 * 处理待转换的任务
 */
async function processPendingConversions() {
  try {
    // 获取所有转换，然后过滤出 pending 状态的
    const allConversions = await getUserConversions(1, 100); // 获取足够多的任务
    const pendingConversions = allConversions.filter(
      c => c.status === "pending" && !processingConversions.has(c.id)
    );

    for (const conversion of pendingConversions) {
      processingConversions.add(conversion.id);
      await processConversion(conversion);
      processingConversions.delete(conversion.id);
    }
  } catch (error) {
    console.error("[ConversionWorker] Failed to process pending conversions:", error);
  }
}

/**
 * 处理单个转换任务
 */
async function processConversion(conversion: any) {
  try {
    console.log(`[ConversionWorker] Processing conversion ${conversion.id}`);
    
    // 验证转换是否支持
    if (!isConversionSupported(conversion.sourceFormat, conversion.targetFormat)) {
      throw new Error(
        `不支持从 ${conversion.sourceFormat} 转换到 ${conversion.targetFormat}`
      );
    }

    // 从内存中获取上传的文件
    let sourceFileBuffer: Buffer | undefined = uploadedFiles.get(conversion.id);
    
    if (!sourceFileBuffer) {
      // 尝试从文件系统读取
      const sourceFileName = conversion.sourceFileKey.replace(/\//g, "_");
      const sourceFilePath = path.join(STORAGE_DIR, sourceFileName);
      
      try {
        sourceFileBuffer = await fs.promises.readFile(sourceFilePath);
      } catch (error) {
        console.error(`[ConversionWorker] Failed to read source file:`, error);
        throw new Error("源文件不存在");
      }
    }

    // 执行转换
    const result = await convertFile(
      conversion.id,
      sourceFileBuffer,
      conversion.sourceFormat,
      conversion.targetFormat
    );

    // 清理内存中的文件
    uploadedFiles.delete(conversion.id);

    if (!result.success) {
      console.error(
        `[ConversionWorker] Conversion failed for ID ${conversion.id}:`,
        result.error
      );
    } else {
      console.log(`[ConversionWorker] Conversion completed for ID ${conversion.id}`);
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    console.error(
      `[ConversionWorker] Error processing conversion ${conversion.id}:`,
      errorMessage
    );

    // 清理内存中的文件
    uploadedFiles.delete(conversion.id);

    // 更新任务状态为失败
    await updateConversionStatus(conversion.id, "failed", {
      errorMessage,
    });
  }
}
