import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { faceAPI, User } from './face-api';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signUp: (email: string, password: string, fullName: string) => Promise<{ error: Error | null }>;
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>;
  signOut: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const USER_STORAGE_KEY = 'face_attend_user';

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Load user from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(USER_STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setUser(parsed);
      } catch {
        localStorage.removeItem(USER_STORAGE_KEY);
      }
    }
    setLoading(false);
  }, []);

  const signUp = async (email: string, password: string, fullName: string) => {
    try {
      const response = await faceAPI.signup(email, password, fullName);
      setUser(response.user);
      localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(response.user));
      return { error: null };
    } catch (error: any) {
      const message = error.response?.data?.error || 'Sign up failed';
      return { error: new Error(message) };
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      const response = await faceAPI.login(email, password);
      setUser(response.user);
      localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(response.user));
      return { error: null };
    } catch (error: any) {
      const message = error.response?.data?.error || 'Login failed';
      return { error: new Error(message) };
    }
  };

  const signOut = () => {
    setUser(null);
    localStorage.removeItem(USER_STORAGE_KEY);
  };

  return (
    <AuthContext.Provider value={{ user, loading, signUp, signIn, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
