import { Card } from "@/components/ui/card";
import { Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";

interface FileUploadZoneProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
}

const SUPPORTED_FORMATS = [
  "doc", "docx", "xls", "xlsx", "ppt", "pptx",
  "pdf", "jpg", "jpeg", "png", "gif", "bmp"
];

export default function FileUploadZone({
  onFileSelect,
  selectedFile,
}: FileUploadZoneProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): boolean => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!ext || !SUPPORTED_FORMATS.includes(ext)) {
      toast.error(`不支持的文件格式: ${ext}`);
      return false;
    }
    if (file.size > 100 * 1024 * 1024) {
      toast.error("文件大小不能超过 100MB");
      return false;
    }
    return true;
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      if (validateFile(files[0])) {
        onFileSelect(files[0]);
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      if (validateFile(files[0])) {
        onFileSelect(files[0]);
      }
    }
  };

  const handleClick = () => {
    inputRef.current?.click();
  };

  if (selectedFile) {
    return (
      <Card className="p-8 border-2 border-accent/20 bg-accent/5">
        <div className="space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3 flex-1">
              <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center flex-shrink-0">
                <Upload className="w-6 h-6 text-accent" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="font-medium text-foreground truncate">
                  {selectedFile.name}
                </p>
                <p className="text-sm text-muted-foreground">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            <button
              onClick={() => onFileSelect(null as any)}
              className="ml-2 p-2 hover:bg-red-50 dark:hover:bg-red-950 rounded-lg transition-colors text-muted-foreground hover:text-destructive"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      className={`p-12 border-2 border-dashed transition-all cursor-pointer ${
        isDragActive
          ? "border-accent bg-accent/5"
          : "border-border hover:border-accent/50 hover:bg-accent/2"
      }`}
      onClick={handleClick}
    >
      <input
        ref={inputRef}
        type="file"
        onChange={handleChange}
        className="hidden"
        accept={SUPPORTED_FORMATS.map((f) => `.${f}`).join(",")}
      />

      <div className="flex flex-col items-center justify-center space-y-4">
        <div className="w-16 h-16 bg-accent/10 rounded-2xl flex items-center justify-center">
          <Upload className="w-8 h-8 text-accent" />
        </div>

        <div className="text-center space-y-2">
          <p className="text-lg font-semibold text-foreground">
            拖拽文件到这里
          </p>
          <p className="text-sm text-muted-foreground">
            或点击选择文件
          </p>
        </div>

        <p className="text-xs text-muted-foreground">
          支持 Word、Excel、PowerPoint、PDF、图片等格式，最大 100MB
        </p>
      </div>
    </Card>
  );
}
