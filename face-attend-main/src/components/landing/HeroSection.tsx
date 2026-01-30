import { motion, useMotionValue, useTransform, useSpring } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Scan, Shield, Users, Sparkles, Zap, Brain } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useRef, useEffect, useState } from 'react';

// Floating particle component
const FloatingParticle = ({ delay, duration, size, left, color }: { 
  delay: number; 
  duration: number; 
  size: number; 
  left: string;
  color: string;
}) => (
  <motion.div
    className="absolute rounded-full pointer-events-none"
    style={{ 
      width: size, 
      height: size, 
      left,
      bottom: '-10%',
      background: color,
      filter: 'blur(1px)',
    }}
    initial={{ y: 0, opacity: 0 }}
    animate={{ 
      y: [0, -1200], 
      opacity: [0, 1, 1, 0],
      scale: [0.5, 1, 1, 0.5],
    }}
    transition={{ 
      duration, 
      delay, 
      repeat: Infinity,
      ease: "linear"
    }}
  />
);

// 3D Face mesh SVG component
const Face3DMesh = () => {
  return (
    <motion.div 
      className="relative w-64 h-64 md:w-80 md:h-80"
      animate={{ rotateY: [0, 360] }}
      transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
    >
      <svg viewBox="0 0 200 200" className="w-full h-full">
        {/* Outer glow */}
        <defs>
          <radialGradient id="faceGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="hsl(195, 100%, 50%)" stopOpacity="0.8" />
            <stop offset="50%" stopColor="hsl(226, 70%, 55%)" stopOpacity="0.4" />
            <stop offset="100%" stopColor="hsl(280, 80%, 60%)" stopOpacity="0" />
          </radialGradient>
          <filter id="neonGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feFlood floodColor="hsl(195, 100%, 50%)" result="color" />
            <feComposite in="color" in2="blur" operator="in" result="glow" />
            <feMerge>
              <feMergeNode in="glow" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        
        {/* Background glow circle */}
        <circle cx="100" cy="100" r="90" fill="url(#faceGlow)" opacity="0.5" />
        
        {/* Face mesh wireframe */}
        <g fill="none" stroke="hsl(195, 100%, 50%)" strokeWidth="0.5" filter="url(#neonGlow)" opacity="0.8">
          {/* Face outline */}
          <ellipse cx="100" cy="100" rx="60" ry="75" />
          
          {/* Horizontal lines */}
          {[30, 50, 70, 90, 110, 130, 150, 170].map((y, i) => (
            <motion.path 
              key={`h-${i}`}
              d={`M ${40 + Math.abs(100 - y) * 0.3} ${y} Q 100 ${y + (Math.sin(y/30) * 5)} ${160 - Math.abs(100 - y) * 0.3} ${y}`}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 2, delay: i * 0.1 }}
            />
          ))}
          
          {/* Vertical lines */}
          {[50, 70, 90, 100, 110, 130, 150].map((x, i) => (
            <motion.path 
              key={`v-${i}`}
              d={`M ${x} ${25 + Math.abs(100 - x) * 0.5} Q ${x + (Math.sin(x/30) * 3)} 100 ${x} ${175 - Math.abs(100 - x) * 0.5}`}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 2, delay: 0.5 + i * 0.1 }}
            />
          ))}
          
          {/* Eyes */}
          <ellipse cx="75" cy="85" rx="12" ry="6" strokeWidth="1" />
          <ellipse cx="125" cy="85" rx="12" ry="6" strokeWidth="1" />
          <circle cx="75" cy="85" r="3" fill="hsl(195, 100%, 50%)" />
          <circle cx="125" cy="85" r="3" fill="hsl(195, 100%, 50%)" />
          
          {/* Nose */}
          <path d="M 100 90 L 95 115 L 100 120 L 105 115 L 100 90" strokeWidth="0.8" />
          
          {/* Mouth */}
          <path d="M 80 140 Q 100 150 120 140" strokeWidth="1" />
          
          {/* Face detection points */}
          {[
            [100, 25], [60, 50], [140, 50], [40, 100], [160, 100],
            [60, 150], [140, 150], [100, 175], [75, 85], [125, 85],
            [100, 120], [90, 140], [110, 140]
          ].map(([x, y], i) => (
            <motion.circle 
              key={`point-${i}`}
              cx={x} 
              cy={y} 
              r="3"
              fill="hsl(195, 100%, 50%)"
              initial={{ scale: 0 }}
              animate={{ scale: [0, 1.5, 1] }}
              transition={{ duration: 0.5, delay: 1 + i * 0.05 }}
            />
          ))}
        </g>
        
        {/* Scanning line animation */}
        <motion.rect 
          x="35" 
          y="20" 
          width="130" 
          height="3" 
          fill="url(#scanGradient)"
          initial={{ y: 20 }}
          animate={{ y: [20, 180, 20] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        />
        <defs>
          <linearGradient id="scanGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="transparent" />
            <stop offset="50%" stopColor="hsl(195, 100%, 50%)" />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
        </defs>
      </svg>
      
      {/* Orbiting elements */}
      {[0, 120, 240].map((angle, i) => (
        <motion.div
          key={i}
          className="absolute w-4 h-4 rounded-full bg-gradient-to-r from-cyan-400 to-purple-500"
          style={{
            top: '50%',
            left: '50%',
            marginTop: -8,
            marginLeft: -8,
          }}
          animate={{
            rotate: [angle, angle + 360],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "linear",
          }}
        >
          <motion.div
            className="w-4 h-4 rounded-full bg-gradient-to-r from-cyan-400 to-purple-500"
            style={{
              transform: 'translateX(130px)',
              boxShadow: '0 0 20px hsl(195, 100%, 50%)',
            }}
          />
        </motion.div>
      ))}
    </motion.div>
  );
};

// Stats counter component
const AnimatedCounter = ({ value, suffix = '' }: { value: number; suffix?: string }) => {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const stepValue = value / steps;
    let current = 0;
    
    const timer = setInterval(() => {
      current += stepValue;
      if (current >= value) {
        setCount(value);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, duration / steps);
    
    return () => clearInterval(timer);
  }, [value]);
  
  return <span>{count.toLocaleString()}{suffix}</span>;
};

export function HeroSection() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  
  const rotateX = useSpring(useTransform(mouseY, [-300, 300], [10, -10]), { stiffness: 100, damping: 30 });
  const rotateY = useSpring(useTransform(mouseX, [-300, 300], [-10, 10]), { stiffness: 100, damping: 30 });

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      mouseX.set(e.clientX - rect.left - rect.width / 2);
      mouseY.set(e.clientY - rect.top - rect.height / 2);
    }
  };

  return (
    <section 
      ref={containerRef}
      onMouseMove={handleMouseMove}
      className="relative min-h-screen flex items-center justify-center overflow-hidden px-4 py-20"
    >
      {/* Animated mesh gradient background */}
      <div className="absolute inset-0 -z-10 gradient-mesh" />
      
      {/* Aurora effect */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <motion.div 
          className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-primary/30 via-purple-500/20 to-transparent rounded-full blur-3xl"
          animate={{ 
            rotate: [0, 360],
            scale: [1, 1.2, 1],
          }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        />
        <motion.div 
          className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-accent/30 via-pink-500/20 to-transparent rounded-full blur-3xl"
          animate={{ 
            rotate: [360, 0],
            scale: [1, 1.3, 1],
          }}
          transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
        />
      </div>
      
      {/* Cyber grid */}
      <div className="absolute inset-0 -z-10 cyber-grid opacity-30" />
      
      {/* Floating particles */}
      <div className="absolute inset-0 -z-5 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <FloatingParticle
            key={i}
            delay={i * 0.5}
            duration={10 + Math.random() * 10}
            size={4 + Math.random() * 8}
            left={`${Math.random() * 100}%`}
            color={`hsl(${195 + Math.random() * 85}, 100%, ${50 + Math.random() * 20}%)`}
          />
        ))}
      </div>
      
      <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
        {/* Left side - Content */}
        <div className="text-center lg:text-left z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-6"
          >
            <span className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-primary/20 via-purple-500/20 to-pink-500/20 border border-primary/30 text-primary text-sm font-semibold backdrop-blur-sm shadow-lg shadow-primary/10">
              <Sparkles className="w-4 h-4 animate-pulse" />
              <span className="text-gradient-cyber">AI-Powered Facial Recognition</span>
              <Zap className="w-4 h-4 text-yellow-400" />
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-5xl md:text-7xl lg:text-8xl font-bold font-display mb-6 tracking-tight leading-[0.9]"
          >
            <span className="block">Smart</span>
            <span className="block text-gradient-neon">Attendance</span>
            <span className="block bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent">
              Reimagined
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-xl md:text-2xl text-muted-foreground max-w-xl mx-auto lg:mx-0 mb-10 leading-relaxed"
          >
            Experience the future of attendance tracking. 
            <span className="text-primary font-semibold"> Instant recognition</span>, 
            <span className="text-purple-500 font-semibold"> real-time monitoring</span>, 
            <span className="text-pink-500 font-semibold"> beautiful analytics</span>.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start mb-12"
          >
            <Link to="/signup">
              <Button variant="hero" size="xl" className="w-full sm:w-auto group">
                <span className="relative z-10 flex items-center gap-2">
                  Get Started Free
                  <motion.span
                    animate={{ x: [0, 5, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    →
                  </motion.span>
                </span>
              </Button>
            </Link>
            <Link to="/login">
              <Button variant="neon" size="xl" className="w-full sm:w-auto">
                Sign In
              </Button>
            </Link>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="grid grid-cols-3 gap-6"
          >
            {[
              { value: 99.9, suffix: '%', label: 'Accuracy' },
              { value: 500, suffix: 'ms', label: 'Response' },
              { value: 10000, suffix: '+', label: 'Users' },
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                className="text-center lg:text-left p-4 rounded-2xl bg-card/50 backdrop-blur-sm border border-border/50 hover:border-primary/30 transition-colors"
                whileHover={{ scale: 1.05, y: -5 }}
              >
                <div className="text-2xl md:text-3xl font-bold font-cyber text-gradient">
                  <AnimatedCounter value={stat.value} suffix={stat.suffix} />
                </div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* Right side - 3D Face Animation */}
        <motion.div
          className="relative flex items-center justify-center"
          style={{ 
            perspective: 1000,
            rotateX,
            rotateY,
          }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <div className="relative">
            {/* Glow background */}
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/30 via-purple-500/30 to-pink-500/30 rounded-full blur-3xl animate-pulse" />
            
            {/* 3D Face mesh */}
            <motion.div
              className="relative z-10"
              animate={{ 
                y: [0, -10, 0],
              }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            >
              <Face3DMesh />
            </motion.div>
            
            {/* Floating icons */}
            {[
              { Icon: Brain, delay: 0, x: -120, y: -80 },
              { Icon: Shield, delay: 0.5, x: 120, y: -60 },
              { Icon: Users, delay: 1, x: -100, y: 80 },
              { Icon: Scan, delay: 1.5, x: 100, y: 100 },
            ].map(({ Icon, delay, x, y }, i) => (
              <motion.div
                key={i}
                className="absolute w-12 h-12 rounded-xl bg-card/80 backdrop-blur-sm border border-border/50 flex items-center justify-center shadow-lg"
                style={{ left: '50%', top: '50%' }}
                initial={{ opacity: 0, x: 0, y: 0 }}
                animate={{ 
                  opacity: 1, 
                  x,
                  y: [y, y - 10, y],
                }}
                transition={{ 
                  opacity: { delay: 1 + delay, duration: 0.5 },
                  x: { delay: 1 + delay, duration: 0.5 },
                  y: { delay: 1.5 + delay, duration: 3, repeat: Infinity },
                }}
              >
                <Icon className="w-6 h-6 text-primary" />
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
      
      {/* Scroll indicator */}
      <motion.div 
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
      >
        <motion.div
          className="w-6 h-10 rounded-full border-2 border-primary/50 flex justify-center p-2"
          animate={{ y: [0, 5, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        >
          <motion.div 
            className="w-1.5 h-1.5 rounded-full bg-primary"
            animate={{ y: [0, 12, 0], opacity: [1, 0.5, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        </motion.div>
      </motion.div>
    </section>
  );
}
