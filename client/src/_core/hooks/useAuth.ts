import { useCallback, useMemo } from "react";
import { trpc } from "@/lib/trpc";

type User = {
  id: number;
  name?: string | null;
  openId?: string | null;
  email?: string | null;
  role?: string | null;
};

type UseAuthOptions = {
  redirectOnUnauthenticated?: boolean;
  redirectPath?: string;
};

export function useAuth(_options?: UseAuthOptions) {
  const meQuery = trpc.auth.me.useQuery(undefined, {
    // keep it polite: let the hook run on mount
    retry: false,
  });

  const loginMut = trpc.auth.login.useMutation();
  const registerMut = trpc.auth.register.useMutation();
  const logoutMut = trpc.auth.logout.useMutation();

  const user = meQuery.data as unknown as User | null;
  const loading = meQuery.isLoading;

  const error = (
    meQuery.error?.message || loginMut.error?.message || registerMut.error?.message || undefined
  ) as string | undefined | null;

  const isAuthenticated = Boolean(user && (user as any).id);

  const login = useCallback(async (username: string, password: string) => {
    try {
      await loginMut.mutateAsync({ username, password });
      await meQuery.refetch();
      return true;
    } catch (e) {
      return false;
    }
  }, [loginMut, meQuery]);

  const register = useCallback(async (username: string, _email: string, password: string) => {
    try {
      await registerMut.mutateAsync({ username, password });
      await meQuery.refetch();
      return true;
    } catch (e) {
      return false;
    }
  }, [registerMut, meQuery]);

  const logout = useCallback(async () => {
    try {
      await logoutMut.mutateAsync();
      await meQuery.refetch();
    } catch (e) {
      // ignore
    }
  }, [logoutMut, meQuery]);

  const refresh = useCallback(() => {
    meQuery.refetch();
  }, [meQuery]);

  return useMemo(() => ({
    user,
    loading,
    error,
    isAuthenticated,
    login,
    register,
    logout,
    refresh,
  }), [user, loading, error, isAuthenticated, login, register, logout, refresh]);
}
