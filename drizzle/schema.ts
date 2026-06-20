import { int, mysqlEnum, mysqlTable, text, timestamp, varchar } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 * Extend this file with additional tables as your product grows.
 * Columns use camelCase to match both database fields and generated types.
 */
export const users = mysqlTable("users", {
  /**
   * Surrogate primary key. Auto-incremented numeric value managed by the database.
   * Use this for relations between tables.
   */
  id: int("id").autoincrement().primaryKey(),
  /** Manus OAuth identifier (openId) returned from the OAuth callback. Unique per user. */
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  /** Hashed password for local username/password login */
  passwordHash: varchar("passwordHash", { length: 255 }),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * 转换任务表：记录每次文件转换的任务信息
 */
export const conversions = mysqlTable("conversions", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull().references(() => users.id),
  sourceFileName: varchar("sourceFileName", { length: 255 }).notNull(),
  sourceFormat: varchar("sourceFormat", { length: 50 }).notNull(), // e.g., 'docx', 'xlsx', 'pdf'
  targetFormat: varchar("targetFormat", { length: 50 }).notNull(), // e.g., 'pdf', 'docx', 'jpg'
  status: mysqlEnum("status", ["pending", "converting", "completed", "failed"]).default("pending").notNull(),
  errorMessage: text("errorMessage"),
  sourceFileKey: varchar("sourceFileKey", { length: 255 }).notNull(), // S3 key
  sourceFileUrl: varchar("sourceFileUrl", { length: 512 }).notNull(), // S3 URL
  resultFileKey: varchar("resultFileKey", { length: 255 }), // S3 key for result
  resultFileUrl: varchar("resultFileUrl", { length: 512 }), // S3 URL for result
  resultFileName: varchar("resultFileName", { length: 255 }), // Generated result filename
  fileSize: int("fileSize"), // Source file size in bytes
  resultFileSize: int("resultFileSize"), // Result file size in bytes
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  completedAt: timestamp("completedAt"),
});

export type Conversion = typeof conversions.$inferSelect;
export type InsertConversion = typeof conversions.$inferInsert;