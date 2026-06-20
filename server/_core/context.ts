import type { CreateExpressContextOptions } from "@trpc/server/adapters/express";
import type { User } from "../../drizzle/schema";
import { sdk } from "./sdk";

export type TrpcContext = {
  req: CreateExpressContextOptions["req"];
  res: CreateExpressContextOptions["res"];
  user: User | null;
};

// 模拟用户数据
const mockUser: User = {
  id: 1,
  openId: "mock-openid-1",
  name: "测试用户",
  email: "test@example.com",
  loginMethod: "mock",
  role: "user",
  passwordHash: "",
  createdAt: new Date(),
  updatedAt: new Date(),
  lastSignedIn: new Date(),
};

export async function createContext(
  opts: CreateExpressContextOptions
): Promise<TrpcContext> {
  let user: User | null = null;

  try {
    user = await sdk.authenticateRequest(opts.req);
  } catch (error) {
    // 如果认证失败，使用模拟用户
    user = mockUser;
  }

  // 如果没有认证信息，使用模拟用户
  if (!user) {
    user = mockUser;
  }

  return {
    req: opts.req,
    res: opts.res,
    user,
  };
}
