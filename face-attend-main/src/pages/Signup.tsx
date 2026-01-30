import { useState } from 'react';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Scan, ArrowLeft, Loader2, Eye, EyeOff, Sparkles, Shield, CheckCircle, User, Mail, Lock } from 'lucide-react';
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
      x: [0, -30, 0, 30, 0],
      y: [0, 20, 0, -20, 0],
      scale: [1, 0.9, 1, 1.1, 1],
    }}
    transition={{ 
      duration: 12 + delay * 2, 
      delay, 
      repeat: Infinity,
      ease: "easeInOut"
    }}
  />
);

// Password strength indicator
const PasswordStrength = ({ password }: { password: string }) => {
  const getStrength = () => {
    let strength = 0;
    if (password.length >= 6) strength++;
    if (password.length >= 8) strength++;
    if (/[A-Z]/.test(password)) strength++;
    if (/[0-9]/.test(password)) strength++;
    if (/[^A-Za-z0-9]/.test(password)) strength++;
    return strength;
  };
  
  const strength = getStrength();
  const colors = ['bg-red-500', 'bg-orange-500', 'bg-yellow-500', 'bg-green-400', 'bg-green-500'];
  const labels = ['Weak', 'Fair', 'Good', 'Strong', 'Excellent'];
  
  if (!password) return null;
  
  return (
    <div className="mt-2">
      <div className="flex gap-1 h-1">
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            className={`flex-1 rounded-full ${i < strength ? colors[strength - 1] : 'bg-muted'}`}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: i < strength ? 1 : 1 }}
            transition={{ delay: i * 0.1 }}
          />
        ))}
      </div>
      <p className={`text-xs mt-1 ${strength > 3 ? 'text-green-500' : 'text-muted-foreground'}`}>
        {labels[strength - 1] || 'Too weak'}
      </p>
    </div>
  );
};

export default function Signup() {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { signUp } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    const { error } = await signUp(email, password, fullName);

    if (error) {
      toast({
        variant: 'destructive',
        title: 'Signup failed',
        description: error.message,
      });
      setLoading(false);
      return;
    }

    toast({
      title: 'Account created!',
      description: 'Welcome to FaceAttend.',
    });
    
    navigate('/dashboard');
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4 py-12 overflow-hidden">
      {/* Animated mesh gradient background */}
      <div className="absolute inset-0 -z-10 gradient-mesh" />
      
      {/* Animated background orbs */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <FloatingOrb delay={0} size={400} left="60%" top="15%" color="hsl(280 80% 60% / 0.15)" />
        <FloatingOrb delay={2} size={300} left="20%" top="20%" color="hsl(226 70% 55% / 0.12)" />
        <FloatingOrb delay={4} size={350} left="10%" top="60%" color="hsl(15 90% 60% / 0.1)" />
        <FloatingOrb delay={1} size={250} left="70%" top="70%" color="hsl(142 72% 45% / 0.12)" />
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
          <div className="absolute -inset-[1px] rounded-3xl bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 opacity-20 group-hover:opacity-40 blur transition-opacity duration-500" />
          <div className="absolute -inset-[1px] rounded-3xl bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300" style={{ padding: '1px' }}>
            <div className="w-full h-full rounded-3xl bg-card" />
          </div>
          
          <Card className="relative shadow-2xl border-border/50 backdrop-blur-xl bg-card/90 rounded-3xl overflow-hidden">
            {/* Top gradient line */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500" />
            
            <CardHeader className="text-center pt-10 pb-6">
              <motion.div 
                className="mx-auto w-20 h-20 rounded-2xl bg-gradient-to-br from-purple-500 via-pink-500 to-orange-500 flex items-center justify-center mb-6 shadow-lg shadow-purple-500/30"
                animate={{ 
                  rotateY: [0, -10, 0, 10, 0],
                }}
                transition={{ duration: 5, repeat: Infinity }}
                whileHover={{ scale: 1.1, rotate: -5 }}
              >
                <Scan className="w-10 h-10 text-white" />
                
                {/* Pulse ring */}
                <motion.div
                  className="absolute inset-0 rounded-2xl border-2 border-pink-400/50"
                  animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </motion.div>
              
              <CardTitle className="text-3xl font-display font-bold">
                <span className="bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 bg-clip-text text-transparent">
                  Join FaceAttend
                </span>
              </CardTitle>
              <CardDescription className="text-base mt-2">
                Create your account in seconds
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
                  <Label htmlFor="name" className="text-sm font-semibold">Full Name</Label>
                  <div className="relative">
                    <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                    <Input
                      id="name"
                      type="text"
                      placeholder="John Doe"
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      required
                      className="h-12 pl-12 pr-4 rounded-xl border-2 border-border/50 bg-background/50 backdrop-blur-sm focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                    />
                  </div>
                </motion.div>
                
                <motion.div 
                  className="space-y-2"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <Label htmlFor="email" className="text-sm font-semibold">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      className="h-12 pl-12 pr-4 rounded-xl border-2 border-border/50 bg-background/50 backdrop-blur-sm focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                    />
                  </div>
                </motion.div>
                
                <motion.div 
                  className="space-y-2"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <Label htmlFor="password" className="text-sm font-semibold">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                    <Input
                      id="password"
                      type={showPassword ? 'text' : 'password'}
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      minLength={6}
                      className="h-12 pl-12 pr-12 rounded-xl border-2 border-border/50 bg-background/50 backdrop-blur-sm focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                  <PasswordStrength password={password} />
                </motion.div>
                
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  <Button
                    type="submit"
                    variant="accent"
                    className="w-full h-12 text-base bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500"
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Creating account...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5" />
                        Create Account
                      </>
                    )}
                  </Button>
                </motion.div>
              </form>

              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 }}
              >
                {/* Benefits */}
                <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-purple-500/5 via-pink-500/5 to-orange-500/5 border border-purple-500/10">
                  <p className="text-xs font-medium text-muted-foreground mb-3">What you'll get:</p>
                  <div className="grid grid-cols-2 gap-2">
                    {['Face Recognition', 'Real-time Tracking', 'Analytics', 'Free Forever'].map((item, i) => (
                      <motion.div
                        key={item}
                        className="flex items-center gap-2 text-xs text-muted-foreground"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.7 + i * 0.1 }}
                      >
                        <CheckCircle className="w-3 h-3 text-green-500" />
                        {item}
                      </motion.div>
                    ))}
                  </div>
                </div>
                
                <p className="text-center text-sm text-muted-foreground mt-6">
                  Already have an account?{' '}
                  <Link to="/login" className="text-purple-500 hover:text-purple-400 font-semibold transition-colors">
                    Sign in
                  </Link>
                </p>
              </motion.div>
              
              {/* Security badge */}
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8 }}
                className="mt-6 flex items-center justify-center gap-2 text-xs text-muted-foreground"
              >
                <Shield className="w-4 h-4 text-green-500" />
                <span>Your data is encrypted and secure</span>
              </motion.div>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
    </div>
  );
}
