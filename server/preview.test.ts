import { describe, expect, it, beforeEach, vi } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";
import type { User } from "@shared/types";

// Mock storageGetSignedUrl
vi.mock("./storage", () => ({
  storageGetSignedUrl: vi.fn().mockResolvedValue("https://signed-url.example.com/file.pdf"),
}));

// Mock database functions
vi.mock("./db", async () => {
  const actual = await vi.importActual<typeof import("./db")>("./db");
  return {
    ...actual,
    getConversionById: vi.fn(),
  };
});

function createMockContext(userId: number = 1): TrpcContext {
  const user: User = {
    id: userId,
    openId: "test-user",
    email: "test@example.com",
    name: "Test User",
    loginMethod: "test",
    role: "user",
    createdAt: new Date(),
    updatedAt: new Date(),
    lastSignedIn: new Date(),
  };

  return {
    user,
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {
      clearCookie: vi.fn(),
    } as unknown as TrpcContext["res"],
  };
}

describe("conversion.getPreviewUrl", () => {
  it("should return preview URL for completed conversion", async () => {
    const ctx = createMockContext();
    const caller = appRouter.createCaller(ctx);

    // Mock the database response
    const { getConversionById } = await import("./db");
    vi.mocked(getConversionById).mockResolvedValueOnce({
      id: 1,
      userId: ctx.user.id,
      sourceFileName: "test.docx",
      sourceFormat: "docx",
      targetFormat: "pdf",
      status: "completed",
      sourceFileKey: "conversions/1/test.docx",
      sourceFileUrl: "/manus-storage/conversions/1/test.docx",
      resultFileKey: "conversions/1/result.pdf",
      resultFileUrl: "/manus-storage/conversions/1/result.pdf",
      resultFileName: "result.pdf",
      fileSize: 1024,
      resultFileSize: 2048,
      errorMessage: null,
      createdAt: new Date(),
      updatedAt: new Date(),
      completedAt: new Date(),
    } as any);

    const result = await caller.conversion.getPreviewUrl({ conversionId: 1 });

    expect(result).toBeDefined();
    expect(result.format).toBe("pdf");
    expect(result.fileName).toBe("test.docx");
    expect(result.url).toBe("https://signed-url.example.com/file.pdf");
  });

  it("should throw error if conversion not found", async () => {
    const ctx = createMockContext();
    const caller = appRouter.createCaller(ctx);

    const { getConversionById } = await import("./db");
    vi.mocked(getConversionById).mockResolvedValueOnce(null);

    try {
      await caller.conversion.getPreviewUrl({ conversionId: 999 });
      expect.fail("Should have thrown error");
    } catch (error: any) {
      expect(error.code).toBe("NOT_FOUND");
      expect(error.message).toContain("转换任务不存在");
    }
  });

  it("should throw error if user is not the owner", async () => {
    const ctx = createMockContext(1);
    const caller = appRouter.createCaller(ctx);

    const { getConversionById } = await import("./db");
    vi.mocked(getConversionById).mockResolvedValueOnce({
      id: 1,
      userId: 999, // Different user
      sourceFileName: "test.docx",
      sourceFormat: "docx",
      targetFormat: "pdf",
      status: "completed",
      sourceFileKey: "conversions/999/test.docx",
      sourceFileUrl: "/manus-storage/conversions/999/test.docx",
      resultFileKey: "conversions/999/result.pdf",
      resultFileUrl: "/manus-storage/conversions/999/result.pdf",
      resultFileName: "result.pdf",
      fileSize: 1024,
      resultFileSize: 2048,
      errorMessage: null,
      createdAt: new Date(),
      updatedAt: new Date(),
      completedAt: new Date(),
    } as any);

    try {
      await caller.conversion.getPreviewUrl({ conversionId: 1 });
      expect.fail("Should have thrown error");
    } catch (error: any) {
      expect(error.code).toBe("NOT_FOUND");
    }
  });

  it("should throw error if conversion is not completed", async () => {
    const ctx = createMockContext();
    const caller = appRouter.createCaller(ctx);

    const { getConversionById } = await import("./db");
    vi.mocked(getConversionById).mockResolvedValueOnce({
      id: 1,
      userId: ctx.user.id,
      sourceFileName: "test.docx",
      sourceFormat: "docx",
      targetFormat: "pdf",
      status: "pending", // Not completed
      sourceFileKey: "conversions/1/test.docx",
      sourceFileUrl: "/manus-storage/conversions/1/test.docx",
      resultFileKey: null,
      resultFileUrl: null,
      resultFileName: null,
      fileSize: 1024,
      resultFileSize: null,
      errorMessage: null,
      createdAt: new Date(),
      updatedAt: new Date(),
      completedAt: null,
    } as any);

    try {
      await caller.conversion.getPreviewUrl({ conversionId: 1 });
      expect.fail("Should have thrown error");
    } catch (error: any) {
      expect(error.code).toBe("BAD_REQUEST");
      expect(error.message).toContain("未完成");
    }
  });

  it("should throw error if result file key is missing", async () => {
    const ctx = createMockContext();
    const caller = appRouter.createCaller(ctx);

    const { getConversionById } = await import("./db");
    vi.mocked(getConversionById).mockResolvedValueOnce({
      id: 1,
      userId: ctx.user.id,
      sourceFileName: "test.docx",
      sourceFormat: "docx",
      targetFormat: "pdf",
      status: "completed",
      sourceFileKey: "conversions/1/test.docx",
      sourceFileUrl: "/manus-storage/conversions/1/test.docx",
      resultFileKey: null, // Missing
      resultFileUrl: "/manus-storage/conversions/1/result.pdf",
      resultFileName: "result.pdf",
      fileSize: 1024,
      resultFileSize: 2048,
      errorMessage: null,
      createdAt: new Date(),
      updatedAt: new Date(),
      completedAt: new Date(),
    } as any);

    try {
      await caller.conversion.getPreviewUrl({ conversionId: 1 });
      expect.fail("Should have thrown error");
    } catch (error: any) {
      expect(error.code).toBe("BAD_REQUEST");
    }
  });
});
