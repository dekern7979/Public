import { COOKIE_NAME, ONE_YEAR_MS } from "@shared/const";
import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { getSessionCookieOptions } from "./_core/cookies";
import { sdk } from "./_core/sdk";
import { systemRouter } from "./_core/systemRouter";
import { protectedProcedure, publicProcedure, router } from "./_core/trpc";
import { createConversion, getConversionById, getUserConversions, updateConversionStatus } from "./db";
import { storagePut, storageGetSignedUrl } from "./storage";
import { storeUploadedFile } from "./conversion-worker";

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    login: publicProcedure
      .input(z.object({
        username: z.string().min(1, "请输入用户名"),
        password: z.string().min(1, "请输入密码"),
      }))
      .mutation(async ({ ctx, input }) => {
        try {
          const { user, token } = await sdk.loginLocal(input.username, input.password);
          const cookieOptions = getSessionCookieOptions(ctx.req);
          ctx.res.cookie(COOKIE_NAME, token, { ...cookieOptions, maxAge: ONE_YEAR_MS });
          return { user };
        } catch (error) {
          throw new TRPCError({
            code: "UNAUTHORIZED",
            message: error instanceof Error ? error.message : "登录失败",
          });
        }
      }),
    register: publicProcedure
      .input(z.object({
        username: z.string().min(2, "用户名至少2个字符").max(32, "用户名最多32个字符"),
        password: z.string().min(4, "密码至少4个字符").max(64, "密码最多64个字符"),
      }))
      .mutation(async ({ ctx, input }) => {
        try {
          const { user, token } = await sdk.registerLocal(input.username, input.password);
          const cookieOptions = getSessionCookieOptions(ctx.req);
          ctx.res.cookie(COOKIE_NAME, token, { ...cookieOptions, maxAge: ONE_YEAR_MS });
          return { user };
        } catch (error) {
          throw new TRPCError({
            code: "CONFLICT",
            message: error instanceof Error ? error.message : "注册失败",
          });
        }
      }),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  conversion: router({
    /**
     * 上传文件并创建转换任务
     */
    upload: protectedProcedure
      .input(z.object({
        fileName: z.string(),
        sourceFormat: z.string(),
        targetFormat: z.string(),
        fileBuffer: z.instanceof(Uint8Array),
      }))
      .mutation(async ({ ctx, input }) => {
        try {
          // 上传文件到 S3
          // 将文件名转换为 ASCII 安全的格式（使用时间戳 + 原始扩展名）
          const fileExt = input.fileName.split('.').pop() || 'bin';
          const safeFileName = `${Date.now()}.${fileExt}`;
          const sourceFileKey = `conversions/${ctx.user.id}/${safeFileName}`;
          const { url: sourceFileUrl } = await storagePut(
            sourceFileKey,
            Buffer.from(input.fileBuffer),
            'application/octet-stream'
          );

          // 创建转换任务记录
          const result = await createConversion({
            userId: ctx.user.id,
            sourceFileName: input.fileName, // 保存原始文件名!
            sourceFormat: input.sourceFormat,
            targetFormat: input.targetFormat,
            status: 'pending',
            sourceFileKey,
            sourceFileUrl,
            fileSize: input.fileBuffer.length,
          });

          const conversionId = (result as any).insertId;

          // 存储文件到内存中供 worker 使用
          storeUploadedFile(conversionId, Buffer.from(input.fileBuffer));

          return {
            conversionId,
            sourceFileUrl,
            originalFileName: input.fileName, // 返回原始文件名供前端显示
          };
        } catch (error) {
          console.error('Upload error:', error);
          throw new TRPCError({
            code: 'INTERNAL_SERVER_ERROR',
            message: '文件上传失败',
          });
        }
      }),

    /**
     * 获取转换历史
     */
    getHistory: protectedProcedure
      .input(z.object({
        limit: z.number().default(20),
        offset: z.number().default(0),
      }))
      .query(async ({ ctx, input }) => {
        const conversions = await getUserConversions(
          ctx.user.id,
          input.limit,
          input.offset
        );
        return conversions;
      }),

    /**
     * 获取转换任务详情
     */
    getById: protectedProcedure
      .input(z.object({ conversionId: z.number() }))
      .query(async ({ ctx, input }) => {
        const conversion = await getConversionById(input.conversionId);
        if (!conversion || conversion.userId !== ctx.user.id) {
          throw new TRPCError({
            code: 'NOT_FOUND',
            message: '转换任务不存在',
          });
        }
        return conversion;
      }),

    /**
     * 获取文件预览 URL
     */
    getPreviewUrl: protectedProcedure
      .input(z.object({ conversionId: z.number() }))
      .query(async ({ ctx, input }) => {
        const conversion = await getConversionById(input.conversionId);
        if (!conversion || conversion.userId !== ctx.user.id) {
          throw new TRPCError({
            code: 'NOT_FOUND',
            message: '转换任务不存在',
          });
        }

        if (!conversion.resultFileKey || conversion.status !== 'completed') {
          throw new TRPCError({
            code: 'BAD_REQUEST',
            message: '转换任务未完成或不存在结果文件',
          });
        }

        try {
          // 获取有效期为 1 小时的签名 URL
          const signedUrl = await storageGetSignedUrl(conversion.resultFileKey);
          return {
            url: signedUrl,
            format: conversion.targetFormat,
            fileName: conversion.sourceFileName,
          };
        } catch (error) {
          console.error('Failed to get preview URL:', error);
          throw new TRPCError({
            code: 'INTERNAL_SERVER_ERROR',
            message: '获取预览 URL 失败',
          });
        }
      }),

    /**
     * 删除转换任务
     */
    delete: protectedProcedure
      .input(z.object({ conversionId: z.number() }))
      .mutation(async ({ ctx, input }) => {
        const conversion = await getConversionById(input.conversionId);
        if (!conversion || conversion.userId !== ctx.user.id) {
          throw new TRPCError({
            code: 'NOT_FOUND',
            message: '转换任务不存在',
          });
        }
        // TODO: 删除 S3 文件
        // TODO: 数据库中标记为已删除或直接删除
        return { success: true };
      }),
  }),
});

export type AppRouter = typeof appRouter;
