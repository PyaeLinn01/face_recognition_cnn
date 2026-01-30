import { useState } from 'react';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Scan, ArrowLeft, Loader2, Eye, EyeOff, Sparkles, Shield } from 'lucide-react';
import { useAuth } from '@/lib/auth-context';
import { useToast } from '@/hooks/use-toast';

// Floating particle component
const FloatingOrb = ({ delay, size, left, top, color }: { 
  delay: number; 
  size: number; 
  left: string;
  top: string;
  color: string;
}) => (
  <motion.div
    className="absolute rounded-full pointer-events-none"
    style={{ 
      width: size, 
      height: size, 
      left,
      top,
      background: color,
      filter: 'blur(40px)',
    }}
    animate={{ 
      x: [0, 30, 0, -30, 0],
      y: [0, -20, 0, 20, 0],
      scale: [1, 1.1, 1, 0.9, 1],
    }}
    transition={{ 
      duration: 10 + delay * 2, 
      delay, 
      repeat: Infinity,
      ease: "easeInOut"
    }}
  />
);

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    const { error } = await signIn(email, password);

    if (error) {
      toast({
        variant: 'destructive',
        title: 'Login failed',
        description: error.message,
      });
      setLoading(false);
      return;
    }

    toast({
      title: 'Welcome back!',
      description: 'You have successfully logged in.',
    });
    
    navigate('/dashboard');
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4 py-12 overflow-hidden">
      {/* Animated mesh gradient background */}
      <div className="absolute inset-0 -z-10 gradient-mesh" />
      
      {/* Animated background orbs */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <FloatingOrb delay={0} size={400} left="10%" top="20%" color="hsl(226 70% 55% / 0.15)" />
        <FloatingOrb delay={2} size={300} left="60%" top="10%" color="hsl(280 80% 60% / 0.12)" />
        <FloatingOrb delay={4} size={350} left="70%" top="60%" color="hsl(320 90% 55% / 0.1)" />
        <FloatingOrb delay={1} size={250} left="20%" top="70%" color="hsl(195 100% 50% / 0.12)" />
      </div>
      
      {/* Cyber grid overlay */}
      <div className="absolute inset-0 -z-10 cyber-grid opacity-20" />

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-md relative z-10"
      >
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-muted-foreground hover:text-primary mb-8 transition-colors group"
          >
            <motion.span
              className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors"
              whileHover={{ x: -3 }}
            >
              <ArrowLeft className="w-4 h-4" />
            </motion.span>
            <span className="font-medium">Back to home</span>
          </Link>
        </motion.div>

        {/* Card with glass effect and gradient border */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="relative group"
        >
          {/* Gradient border glow */}
          <div className="absolute -inset-[1px] rounded-3xl bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 opacity-20 group-hover:opacity-40 blur transition-opacity duration-500" />
          <div className="absolute -inset-[1px] rounded-3xl bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300" style={{ padding: '1px' }}>
            <div className="w-full h-full rounded-3xl bg-card" />
          </div>
          
          <Card className="relative shadow-2xl border-border/50 backdrop-blur-xl bg-card/90 rounded-3xl overflow-hidden">
            {/* Top gradient line */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500" />
            
            <CardHeader className="text-center pt-10 pb-6">
              <motion.div 
                className="mx-auto w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500 via-blue-500 to-purple-500 flex items-center justify-center mb-6 shadow-lg shadow-primary/30"
                animate={{ 
                  rotateY: [0, 10, 0, -10, 0],
                }}
                transition={{ duration: 5, repeat: Infinity }}
                whileHover={{ scale: 1.1, rotate: 5 }}
              >
                <Scan className="w-10 h-10 text-white" />
                
                {/* Pulse ring */}
                <motion.div
                  className="absolute inset-0 rounded-2xl border-2 border-cyan-400/50"
                  animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </motion.div>
              
              <CardTitle className="text-3xl font-display font-bold">
                <span className="text-gradient">Welcome back</span>
              </CardTitle>
              <CardDescription className="text-base mt-2">
                Sign in to continue your journey
              </CardDescription>
            </CardHeader>
            
            <CardContent className="px-8 pb-10">
              <form onSubmit={handleSubmit} className="space-y-5">
                <motion.div 
                  className="space-y-2"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <Label htmlFor="email" className="text-sm font-semibold">Email</Label>
                  <div className="relative">
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      className="h-12 pl-4 pr-4 rounded-xl border-2 border-border/50 bg-background/50 backdrop-blur-sm focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all"
                    />
                  </div>
                </motion.div>
                
                <motion.div 
                  className="space-y-2"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <Label htmlFor="password" className="text-sm font-semibold">Password</Label>
                  <div className="relative">
                    <Input
                      id="password"
                      type={showPassword ? 'text' : 'password'}
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      className="h-12 pl-4 pr-12 rounded-xl border-2 border-border/50 bg-background/50 backdrop-blur-sm focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                </motion.div>
                
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <Button
                    type="submit"
                    variant="hero"
                    className="w-full h-12 text-base"
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Signing in...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5" />
                        Sign In
                      </>
                    )}
                  </Button>
                </motion.div>
              </form>

              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className="mt-8"
              >
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-border/50" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-card px-2 text-muted-foreground">Or</span>
                  </div>
                </div>
                
                <p className="text-center text-sm text-muted-foreground mt-6">
                  Don't have an account?{' '}
                  <Link to="/signup" className="text-primary hover:text-primary/80 font-semibold transition-colors">
                    Sign up for free
                  </Link>
                </p>
              </motion.div>
              
              {/* Security badge */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="mt-8 flex items-center justify-center gap-2 text-xs text-muted-foreground"
              >
                <Shield className="w-4 h-4 text-green-500" />
                <span>Protected with 256-bit SSL encryption</span>
              </motion.div>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
    </div>
  );
}
