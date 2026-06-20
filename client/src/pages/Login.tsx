import LoginForm from "@/components/LoginForm";
import { useLocation } from "wouter";

export default function LoginPage() {
  const [, setLocation] = useLocation();

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <LoginForm onSuccess={() => setLocation("/")} />
    </div>
  );
}
