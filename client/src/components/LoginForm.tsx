import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, User, Lock, Mail } from "lucide-react";
import { useAuth } from "@/_core/hooks/useAuth";

interface LoginFormProps {
  onSuccess: () => void;
}

export default function LoginForm({ onSuccess }: LoginFormProps) {
  const [isLogin, setIsLogin] = useState(true);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("test@example.com");
  const [password, setPassword] = useState("123456");
  const [localError, setLocalError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  
  const { login, register, error: authError, isAuthenticated } = useAuth();

  // 监听 isAuthenticated 变化，自动触发 onSuccess
  useEffect(() => {
    if (isAuthenticated) {
      onSuccess();
    }
  }, [isAuthenticated, onSuccess]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError("");
    setIsLoading(true);

    console.log("提交登录/注册");

    try {
      if (isLogin) {
        // 登录模式
        if (!email.trim() || !password.trim()) {
          setLocalError("请填写邮箱和密码");
          setIsLoading(false);
          return;
        }
        console.log("开始登录", email.trim());
        const success = await login(email.trim(), password);
        console.log("登录结果:", success);
        if (!success) {
          setLocalError(authError || "登录失败，请重试");
        }
      } else {
        // 注册模式
        if (!name.trim() || !email.trim() || !password.trim()) {
          setLocalError("请填写完整信息");
          setIsLoading(false);
          return;
        }
        if (password.length < 4) {
          setLocalError("密码至少4位");
          setIsLoading(false);
          return;
        }
        console.log("开始注册", name.trim(), email.trim());
        const success = await register(name.trim(), email.trim(), password);
        console.log("注册结果:", success);
        if (!success) {
          setLocalError(authError || "注册失败，请重试");
        }
      }
    } catch (err) {
      console.error("操作异常:", err);
      setLocalError("操作失败，请重试");
    } finally {
      setIsLoading(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setLocalError("");
    setName("");
    if (!isLogin) {
      setEmail("test@example.com");
      setPassword("123456");
    }
  };

  const displayError = localError || authError;

  return (
    <Card className="w-full max-w-md mx-auto shadow-xl">
      <CardHeader className="text-center">
        <CardTitle className="text-2xl font-serif">
          {isLogin ? "登录" : "注册"}
        </CardTitle>
        <CardDescription>
          {isLogin 
            ? "欢迎回来，使用邮箱和密码登录" 
            : "创建新账号开始使用文档转换服务"
          }
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* 注册时显示用户名字段 */}
          {!isLogin && (
            <div className="space-y-2">
              <Label htmlFor="name">用户名</Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  id="name"
                  type="text"
                  placeholder="请输入用户名"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="pl-9"
                  disabled={isLoading}
                />
              </div>
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="email">邮箱</Label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                id="email"
                type="email"
                placeholder="请输入邮箱地址"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="pl-9"
                disabled={isLoading}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">密码</Label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                id="password"
                type="password"
                placeholder={isLogin ? "请输入密码" : "请设置密码（至少4位）"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="pl-9"
                autoComplete={isLogin ? "current-password" : "new-password"}
                disabled={isLoading}
              />
            </div>
          </div>

          {displayError && (
            <p className="text-sm text-destructive text-center">{displayError}</p>
          )}

          {/* 快速登录提示 */}
          {isLogin && (
            <div className="bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg text-center">
              <p className="text-xs text-blue-700 dark:text-blue-300">
                💡 快速测试账号: test@example.com / 123456
              </p>
            </div>
          )}

          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                {isLogin ? "登录中..." : "注册中..."}
              </>
            ) : (
              isLogin ? "登录" : "注册"
            )}
          </Button>

          <p className="text-center text-sm text-muted-foreground">
            {isLogin ? "还没有账号？" : "已有账号？"}
            <button
              type="button"
              onClick={toggleMode}
              className="ml-1 text-accent hover:underline font-medium"
              disabled={isLoading}
            >
              {isLogin ? "立即注册" : "去登录"}
            </button>
          </p>
        </form>
      </CardContent>
    </Card>
  );
}
