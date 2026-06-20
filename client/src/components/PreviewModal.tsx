import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Loader2, Download, AlertCircle } from "lucide-react";
import { useState, useEffect } from "react";
import { trpc } from "@/lib/trpc";
import { toast } from "sonner";

interface PreviewUrlData {
  url: string;
  format: string;
  fileName: string;
}

interface PreviewModalProps {
  isOpen: boolean;
  conversionId: number | null;
  onClose: () => void;
}

/**
 * 文件预览模态框组件
 * 支持 PDF 和图片格式的在线预览
 */
export default function PreviewModal({ isOpen, conversionId, onClose }: PreviewModalProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [format, setFormat] = useState<string>("");
  const [fileName, setFileName] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 获取预览 URL
  const { data: previewData, isLoading: isQueryLoading, error: queryError } = trpc.conversion.getPreviewUrl.useQuery(
    { conversionId: conversionId || 0 },
    {
      enabled: isOpen && conversionId !== null,
    }
  );

  // 处理预览数据加载成功
  useEffect(() => {
    if (previewData) {
      setPreviewUrl(previewData.url);
      setFormat(previewData.format);
      setFileName(previewData.fileName);
      setIsLoading(false);
      setError(null);
    }
  }, [previewData]);

  // 处理预览数据加载开始
  useEffect(() => {
    if (isOpen && conversionId !== null) {
      setIsLoading(true);
      setError(null);
    }
  }, [isOpen, conversionId]);

  // 处理查询加载状态
  useEffect(() => {
    if (isQueryLoading) {
      setIsLoading(true);
    }
  }, [isQueryLoading]);

  // 处理查询错误
  useEffect(() => {
    if (queryError) {
      setIsLoading(false);
      const errorMessage = queryError.message || "获取预览失败";
      setError(errorMessage);
      toast.error(errorMessage);
    }
  }, [queryError]);

  const handleDownload = () => {
    if (previewUrl) {
      const link = document.createElement("a");
      link.href = previewUrl;
      link.download = fileName;
      document.body.appendChild(link);
      try {
        link.click();
      } catch (e) {
        // click can occasionally fail; continue to cleanup
      }
      try {
        if (link.parentNode === document.body) {
          document.body.removeChild(link);
        }
      } catch (e) {
        // ignore if already removed by another script/extension
      }
    }
  };

  const handleClose = () => {
    setPreviewUrl(null);
    setFormat("");
    setFileName("");
    setError(null);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] flex flex-col">
        <DialogHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <DialogTitle className="text-lg font-semibold">文件预览</DialogTitle>
          <div className="flex items-center gap-2">
            {previewUrl && (
              <Button
                size="sm"
                variant="outline"
                onClick={handleDownload}
                className="gap-2"
              >
                <Download className="w-4 h-4" />
                下载
              </Button>
            )}
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-auto bg-gray-50 rounded-lg border border-gray-200 flex items-center justify-center min-h-[400px]">
          {isLoading ? (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
              <p className="text-sm text-gray-600">加载预览中...</p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center gap-3 text-center px-6">
              <AlertCircle className="w-8 h-8 text-red-500" />
              <p className="text-red-600 font-medium">预览加载失败</p>
              <p className="text-sm text-gray-600">{error}</p>
              <p className="text-xs text-gray-500 mt-2">请尝试下载文件后使用本地应用打开</p>
            </div>
          ) : previewUrl ? (
            <PreviewContent url={previewUrl} format={format} fileName={fileName} />
          ) : (
            <div className="text-center">
              <p className="text-gray-500">无法加载预览</p>
            </div>
          )}
        </div>

        <div className="text-xs text-gray-500 pt-3 border-t">
          <p>文件: {fileName}</p>
          <p>格式: {format.toUpperCase()}</p>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/**
 * 预览内容组件
 * 根据文件格式显示相应的预览器
 */
function PreviewContent({
  url,
  format,
  fileName,
}: {
  url: string;
  format: string;
  fileName: string;
}) {
  const lowerFormat = format.toLowerCase();
  const [iframeError, setIframeError] = useState(false);
  
  console.log("Preview URL:", url);
  console.log("Format:", lowerFormat);

  // PDF 预览
  if (lowerFormat === "pdf") {
    if (iframeError) {
      return (
        <div className="text-center p-8">
          <p className="text-gray-600 mb-4">PDF 预览加载失败</p>
          <p className="text-sm text-gray-500">请点击上方的「下载」按钮打开文件</p>
        </div>
      );
    }
    return (
      <iframe
        src={url}
        className="w-full h-full rounded-lg border-0"
        title={fileName}
        onError={() => {
          console.log("PDF iframe error");
          setIframeError(true);
        }}
      />
    );
  }

  // 图片预览
  if (["jpg", "jpeg", "png", "gif", "bmp"].includes(lowerFormat)) {
    return (
      <div className="flex items-center justify-center w-full h-full p-4">
        <img
          src={url}
          alt={fileName}
          className="max-w-full max-h-full object-contain rounded-lg shadow-lg"
        />
      </div>
    );
  }

  // 不支持的格式
  return (
    <div className="text-center">
      <p className="text-gray-600 mb-4">暂不支持 {format.toUpperCase()} 格式的在线预览</p>
      <p className="text-sm text-gray-500">请下载文件后使用本地应用打开</p>
    </div>
  );
}
