import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { isConversionSupported } from "./conversion-service";

describe("Conversion Service", () => {
  describe("isConversionSupported", () => {
    it("should support converting to PDF from various formats", () => {
      expect(isConversionSupported("docx", "pdf")).toBe(true);
      expect(isConversionSupported("doc", "pdf")).toBe(true);
      expect(isConversionSupported("xlsx", "pdf")).toBe(true);
      expect(isConversionSupported("pptx", "pdf")).toBe(true);
      expect(isConversionSupported("jpg", "pdf")).toBe(true);
      expect(isConversionSupported("png", "pdf")).toBe(true);
    });

    it("should support converting PDF to images", () => {
      expect(isConversionSupported("pdf", "jpg")).toBe(true);
      expect(isConversionSupported("pdf", "png")).toBe(true);
    });

    it("should support converting PDF to documents", () => {
      expect(isConversionSupported("pdf", "docx")).toBe(true);
      expect(isConversionSupported("pdf", "xlsx")).toBe(true);
      expect(isConversionSupported("pdf", "pptx")).toBe(true);
    });

    it("should not support unsupported conversions", () => {
      expect(isConversionSupported("docx", "xlsx")).toBe(false);
      expect(isConversionSupported("jpg", "png")).toBe(false);
      expect(isConversionSupported("txt", "pdf")).toBe(false);
    });

    it("should be case-insensitive", () => {
      expect(isConversionSupported("DOCX", "PDF")).toBe(true);
      expect(isConversionSupported("Docx", "Pdf")).toBe(true);
    });
  });
});
