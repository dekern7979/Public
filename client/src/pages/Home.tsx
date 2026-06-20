import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Loader2, Upload, Download, History, FileText, ArrowRight } from "lucide-react";
import { useState } from "react";
import { trpc } from "@/lib/trpc";
import FileUploadZone from "@/components/FileUploadZone";
import ConversionHistory from "@/components/ConversionHistory";
import PreviewModal from "@/components/PreviewModal";
import { toast } from "sonner";

export default function Home() {
  const { logout } = useAuth();
  const [activeTab, setActiveTab] = useState<"convert" | "history">("convert");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetFormat, setTargetFormat] = useState("pdf");
  const [isConverting, setIsConverting] = useState(false);
  const [previewModalOpen, setPreviewModalOpen] = useState(false);
  const [selectedConversionId, setSelectedConversionId] = useState<number | null>(null);

  const handlePreview = (conversionId: number) => {
    setSelectedConversionId(conversionId);
    setPreviewModalOpen(true);
  };

  const uploadMutation = trpc.conversion.upload.useMutation();
  const historyQuery = trpc.conversion.getHistory.useQuery(
    { limit: 20, offset: 0 },
    { refetchInterval: 2000, refetchIntervalInBackground: true }
  );

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
  };

  const handleConvert = async () => {
    if (!selectedFile) {
      toast.error("请先选择文件");
      return;
    }

    setIsConverting(true);
    try {
      const buffer = await selectedFile.arrayBuffer();
      const sourceFormat = selectedFile.name.split(".").pop()?.toLowerCase() || "";

      await uploadMutation.mutateAsync({
        fileName: selectedFile.name,
        sourceFormat,
        targetFormat,
        fileBuffer: new Uint8Array(buffer),
      });

      toast.success("文件上传成功，转换中...");
      setSelectedFile(null);
    } catch (error) {
      toast.error("转换失败，请重试");
      console.error(error);
    } finally {
      setIsConverting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-card/70 backdrop-blur-xl border-b border-border/50 shadow-sm">
        <div className="container max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              文档转换器
            </h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container max-w-6xl mx-auto px-4 py-8">
        {/* Tab Navigation */}
        <div className="flex gap-2 mb-8 bg-card/50 backdrop-blur-sm p-1 rounded-xl w-fit mx-auto border border-border/50">
          <button
            onClick={() => setActiveTab("convert")}
            className={`px-6 py-3 rounded-lg font-medium text-sm transition-all duration-300 ${
              activeTab === "convert"
                ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg"
                : "text-muted-foreground hover:text-foreground hover:bg-secondary"
            }`}
          >
            <div className="flex items-center gap-2">
              <Upload className="w-4 h-4" />
              转换文件
            </div>
          </button>
          <button
            onClick={() => setActiveTab("history")}
            className={`px-6 py-3 rounded-lg font-medium text-sm transition-all duration-300 ${
              activeTab === "history"
                ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg"
                : "text-muted-foreground hover:text-foreground hover:bg-secondary"
            }`}
          >
            <div className="flex items-center gap-2">
              <History className="w-4 h-4" />
              转换历史
            </div>
          </button>
        </div>

        {/* Convert Tab */}
        {activeTab === "convert" && (
          <div className="space-y-8">
            <div className="grid lg:grid-cols-5 gap-8">
              {/* Upload Zone */}
              <div className="lg:col-span-3">
                <Card className="p-6 bg-card/80 backdrop-blur-sm border border-border/50 shadow-xl">
                  <FileUploadZone
                    onFileSelect={handleFileSelect}
                    selectedFile={selectedFile}
                  />
                </Card>
              </div>

              {/* Format Selector */}
              <div className="lg:col-span-2 space-y-4">
                <Card className="p-6 bg-card/80 backdrop-blur-sm border border-border/50 shadow-xl">
                  <div className="space-y-6">
                    <div>
                      <label className="text-sm font-semibold text-foreground block mb-3">
                        📄 目标格式
                      </label>
                      <select
                        value={targetFormat}
                        onChange={(e) => setTargetFormat(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-border rounded-xl bg-background text-foreground focus:outline-none focus:border-accent transition-all"
                      >
                        <optgroup label="📚 主要格式">
                          <option value="pdf">PDF</option>
                        </optgroup>
                        <optgroup label="📝 文档格式">
                          <option value="docx">Word (.docx)</option>
                          <option value="doc">Word (.doc)</option>
                          <option value="xlsx">Excel (.xlsx)</option>
                          <option value="xls">Excel (.xls)</option>
                          <option value="pptx">PowerPoint (.pptx)</option>
                          <option value="ppt">PowerPoint (.ppt)</option>
                          <option value="md">Markdown (.md)</option>
                        </optgroup>
                        <optgroup label="🖼️ 图片格式">
                          <option value="jpg">JPG</option>
                          <option value="png">PNG</option>
                        </optgroup>
                      </select>
                    </div>

                    <Button
                      onClick={handleConvert}
                      disabled={!selectedFile || isConverting}
                      className="w-full h-12 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg shadow-blue-500/30 transition-all duration-300 hover:shadow-xl hover:-translate-y-0.5"
                    >
                      {isConverting ? (
                        <>
                          <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                          转换中，请稍候...
                        </>
                      ) : (
                        <>
                          开始转换
                          <ArrowRight className="w-4 h-4 ml-2" />
                        </>
                      )}
                    </Button>

                    <div className="pt-4 border-t border-border/50 space-y-3">
                      <p className="text-sm font-medium text-foreground mb-2">支持的格式：</p>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="text-xs text-muted-foreground bg-secondary/50 px-3 py-1.5 rounded-lg">Word (.doc/.docx)</div>
                        <div className="text-xs text-muted-foreground bg-secondary/50 px-3 py-1.5 rounded-lg">Excel (.xls/.xlsx)</div>
                        <div className="text-xs text-muted-foreground bg-secondary/50 px-3 py-1.5 rounded-lg">PPT (.ppt/.pptx)</div>
                        <div className="text-xs text-muted-foreground bg-secondary/50 px-3 py-1.5 rounded-lg">图片 (.jpg/.png)</div>
                        <div className="text-xs text-muted-foreground bg-secondary/50 px-3 py-1.5 rounded-lg col-span-2">PDF 格式</div>
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
            </div>
          </div>
        )}

        {/* History Tab */}
        {activeTab === "history" && (
          <div className="bg-card/80 backdrop-blur-sm rounded-2xl border border-border/50 shadow-xl p-6">
            <ConversionHistory
              conversions={historyQuery.data || []}
              isLoading={historyQuery.isLoading}
              onPreview={handlePreview}
            />
          </div>
        )}

        {/* Preview Modal 始终挂载 */}
        <PreviewModal
          isOpen={previewModalOpen}
          conversionId={selectedConversionId}
          onClose={() => {
            setPreviewModalOpen(false);
            setSelectedConversionId(null);
          }}
        />
      </main>
    </div>
  );
}
