import { Conversion, InsertConversion, InsertUser } from "../drizzle/schema";

// 内存存储
interface User {
  id: number;
  openId: string;
  name?: string | null;
  email?: string | null;
  loginMethod?: string | null;
  passwordHash?: string | null;
  role: 'user' | 'admin';
  createdAt: Date;
  updatedAt: Date;
  lastSignedIn: Date;
}

const users: Map<string, User> = new Map();
const conversions: Map<number, Conversion> = new Map();
let nextConversionId = 1;

// 初始化默认用户
const defaultUser: User = {
  id: 1,
  openId: "mock-openid-1",
  name: "测试用户",
  email: "test@example.com",
  loginMethod: "mock",
  passwordHash: "",
  role: "user",
  createdAt: new Date(),
  updatedAt: new Date(),
  lastSignedIn: new Date(),
};
users.set(defaultUser.openId, defaultUser);

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const existingUser = users.get(user.openId);
  if (existingUser) {
    existingUser.lastSignedIn = new Date();
    existingUser.updatedAt = new Date();
    if (user.name !== undefined) existingUser.name = user.name;
    if (user.email !== undefined) existingUser.email = user.email;
    users.set(user.openId, existingUser);
  } else {
    const newUser: User = {
      id: users.size + 1,
      openId: user.openId,
      name: user.name ?? null,
      email: user.email ?? null,
      loginMethod: user.loginMethod ?? null,
      passwordHash: user.passwordHash ?? null,
      role: user.role ?? "user",
      createdAt: new Date(),
      updatedAt: new Date(),
      lastSignedIn: user.lastSignedIn ?? new Date(),
    };
    users.set(user.openId, newUser);
  }
}

export async function getUserByOpenId(openId: string) {
  return users.get(openId);
}

/**
 * 获取用户的转换历史记录
 */
export async function getUserConversions(userId: number, limit = 20, offset = 0) {
  const userConversions = Array.from(conversions.values())
    .filter(c => c.userId === userId)
    .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  
  return userConversions.slice(offset, offset + limit);
}

/**
 * 创建转换任务
 */
export async function createConversion(data: InsertConversion) {
  const conversionId = nextConversionId++;
  const conversion: Conversion = {
    id: conversionId,
    ...data,
    createdAt: data.createdAt ?? new Date(),
    updatedAt: data.updatedAt ?? new Date(),
  };
  conversions.set(conversionId, conversion);
  
  return { insertId: conversionId };
}

/**
 * 更新转换任务状态
 */
export async function updateConversionStatus(
  conversionId: number,
  status: "pending" | "converting" | "completed" | "failed",
  updates?: Partial<InsertConversion>
) {
  const conversion = conversions.get(conversionId);
  if (conversion) {
    const updatedConversion: Conversion = {
      ...conversion,
      status,
      ...updates,
      updatedAt: new Date(),
    };
    conversions.set(conversionId, updatedConversion);
  }
}

/**
 * 获取单个转换任务详情
 */
export async function getConversionById(conversionId: number) {
  return conversions.get(conversionId) ?? null;
}


