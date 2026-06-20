import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Download, Loader2, AlertCircle, CheckCircle2, Clock, Eye } from "lucide-react";
import { Conversion } from "@shared/types";
import { format } from "date-fns";
import { zhCN } from "date-fns/locale";

interface ConversionHistoryProps {
  conversions: Conversion[];
  isLoading: boolean;
  onPreview?: (conversionId: number) => void;
}

export default function ConversionHistory({
  conversions,
  isLoading,
  onPreview,
}: ConversionHistoryProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="w-5 h-5 text-green-500" />;
      case "converting":
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
      case "failed":
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Clock className="w-5 h-5 text-amber-500" />;
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case "completed":
        return "已完成";
      case "converting":
        return "转换中";
      case "failed":
        return "失败";
      default:
        return "待处理";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-50 dark:bg-green-950 text-green-700 dark:text-green-300";
      case "converting":
        return "bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300";
      case "failed":
        return "bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-300";
      default:
        return "bg-amber-50 dark:bg-amber-950 text-amber-700 dark:text-amber-300";
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-accent" />
      </div>
    );
  }

  if (conversions.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-gray-500">暂无转换记录</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {conversions.map((conversion) => (
        <Card key={conversion.id} className="hover:shadow-md transition-shadow p-4">
          <div className="flex items-start justify-between">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-3 mb-2">
                {getStatusIcon(conversion.status)}
                <div className="min-w-0 flex-1">
                  <p className="font-medium text-sm truncate">
                    {conversion.sourceFileName}
                  </p>
                  <p className="text-xs text-gray-500">
                    {conversion.sourceFormat.toUpperCase()} → {conversion.targetFormat.toUpperCase()}
                  </p>
                </div>
              </div>
              <p className="text-xs text-gray-400">
                {format(new Date(conversion.createdAt), "PPP p", {
                  locale: zhCN,
                })}
              </p>
              {conversion.fileSize && (
                <p className="text-xs text-gray-400 mt-1">
                  文件大小: {(conversion.fileSize / 1024 / 1024).toFixed(2)} MB
                </p>
              )}
            </div>

            <div className="flex items-center gap-3 ml-4 flex-shrink-0">
              <span
                className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(
                  conversion.status
                )}`}
              >
                {getStatusLabel(conversion.status)}
              </span>

              {conversion.status === "completed" && conversion.resultFileUrl && (
                <>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onPreview?.(conversion.id)}
                    title="预览文件"
                  >
                    <Eye className="w-4 h-4" />
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      const link = document.createElement("a");
                      link.href = conversion.resultFileUrl || "";
                      link.download = conversion.resultFileName || `converted.${conversion.targetFormat}`;
                      document.body.appendChild(link);
                      link.click();
                      try {
                        if (link.parentNode === document.body) {
                          document.body.removeChild(link);
                        }
                      } catch (e) {
                        // ignore if already removed by another script/extension
                      }
                    }}
                    title="下载文件"
                  >
                    <Download className="w-4 h-4" />
                  </Button>
                </>
              )}
            </div>
          </div>

          {conversion.errorMessage && (
            <div className="mt-3 p-3 bg-red-50 dark:bg-red-950 rounded-lg">
              <p className="text-xs text-red-700 dark:text-red-300">
                {conversion.errorMessage}
              </p>
            </div>
          )}
        </Card>
      ))}
    </div>
  );
}
