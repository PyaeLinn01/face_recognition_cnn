import { motion, useMotionTemplate, useMotionValue } from 'framer-motion';
import { 
  Camera, 
  BarChart3, 
  Clock, 
  Users, 
  BookOpen, 
  Shield,
  Zap,
  Fingerprint,
  Sparkles,
  Activity
} from 'lucide-react';
import { useRef } from 'react';

const features = [
  {
    icon: Fingerprint,
    title: 'Face Registration',
    description: 'Students register their face once with our AI-powered system. 4 photos capture your unique biometric profile.',
    color: 'from-cyan-500 to-blue-500',
    shadowColor: 'shadow-cyan-500/20',
    delay: 0,
  },
  {
    icon: Zap,
    title: 'Instant Recognition',
    description: 'Lightning-fast AI recognition marks attendance in under 500ms with 99.9% accuracy.',
    color: 'from-yellow-500 to-orange-500',
    shadowColor: 'shadow-orange-500/20',
    delay: 0.1,
  },
  {
    icon: BarChart3,
    title: 'Analytics Dashboard',
    description: 'Beautiful real-time analytics with attendance percentages, trends, and detailed reports.',
    color: 'from-green-500 to-emerald-500',
    shadowColor: 'shadow-emerald-500/20',
    delay: 0.2,
  },
  {
    icon: Clock,
    title: 'Smart Scheduling',
    description: 'Automatic absence tracking with intelligent scheduling. Never miss a class period.',
    color: 'from-purple-500 to-pink-500',
    shadowColor: 'shadow-purple-500/20',
    delay: 0.3,
  },
  {
    icon: Users,
    title: 'Role-Based Access',
    description: 'Tailored experiences for students, teachers, and admins with custom dashboards.',
    color: 'from-blue-500 to-indigo-500',
    shadowColor: 'shadow-indigo-500/20',
    delay: 0.4,
  },
  {
    icon: Shield,
    title: 'Enterprise Security',
    description: 'Bank-level encryption protects all biometric data. Privacy-first architecture.',
    color: 'from-red-500 to-rose-500',
    shadowColor: 'shadow-rose-500/20',
    delay: 0.5,
  },
];

// Card with mouse follow gradient effect
const FeatureCard = ({ feature, index }: { feature: typeof features[0]; index: number }) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = cardRef.current?.getBoundingClientRect();
    if (rect) {
      mouseX.set(e.clientX - rect.left);
      mouseY.set(e.clientY - rect.top);
    }
  };

  const background = useMotionTemplate`radial-gradient(400px circle at ${mouseX}px ${mouseY}px, hsl(var(--primary) / 0.1), transparent 80%)`;

  return (
    <motion.div
      ref={cardRef}
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, delay: feature.delay }}
      whileHover={{ y: -8, scale: 1.02 }}
      onMouseMove={handleMouseMove}
      className="group relative"
    >
      {/* Gradient border on hover */}
      <div className={`absolute -inset-[1px] rounded-3xl bg-gradient-to-r ${feature.color} opacity-0 group-hover:opacity-100 blur transition-opacity duration-500`} />
      
      <div className="relative h-full p-8 rounded-3xl bg-card border border-border backdrop-blur-sm overflow-hidden">
        {/* Mouse follow gradient */}
        <motion.div
          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
          style={{ background }}
        />
        
        {/* Shimmer effect on hover */}
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
        </div>
        
        {/* Icon */}
        <motion.div 
          className={`relative w-16 h-16 rounded-2xl bg-gradient-to-r ${feature.color} ${feature.shadowColor} shadow-lg flex items-center justify-center mb-6`}
          whileHover={{ rotate: [0, -10, 10, 0], scale: 1.1 }}
          transition={{ duration: 0.5 }}
        >
          <feature.icon className="w-8 h-8 text-white" />
          
          {/* Floating particles around icon */}
          {[...Array(3)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 rounded-full bg-white/50"
              initial={{ opacity: 0 }}
              animate={{ 
                opacity: [0, 1, 0],
                scale: [0, 1, 0],
                x: [0, (i - 1) * 30],
                y: [0, -20 - i * 10],
              }}
              transition={{
                duration: 2,
                delay: i * 0.3,
                repeat: Infinity,
                repeatDelay: 1,
              }}
            />
          ))}
        </motion.div>
        
        {/* Content */}
        <h3 className="relative text-xl font-bold font-display mb-3 group-hover:text-gradient transition-all duration-300">
          {feature.title}
        </h3>
        <p className="relative text-muted-foreground leading-relaxed">
          {feature.description}
        </p>
        
        {/* Learn more link */}
        <motion.div 
          className="relative mt-4 flex items-center gap-2 text-sm font-medium text-primary opacity-0 group-hover:opacity-100 transition-opacity duration-300"
          initial={{ x: -10 }}
          whileHover={{ x: 0 }}
        >
          <span>Learn more</span>
          <motion.span animate={{ x: [0, 5, 0] }} transition={{ duration: 1, repeat: Infinity }}>
            →
          </motion.span>
        </motion.div>
      </div>
    </motion.div>
  );
};

export function FeaturesSection() {
  return (
    <section className="relative py-32 px-4 overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-b from-background via-secondary/30 to-background" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-primary/5 rounded-full blur-3xl" />
        
        {/* Animated grid */}
        <div className="absolute inset-0 cyber-grid opacity-20" />
      </div>
      
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium mb-6"
          >
            <Sparkles className="w-4 h-4" />
            Powerful Features
          </motion.div>
          
          <h2 className="text-4xl md:text-6xl font-bold font-display mb-6">
            Everything You Need,
            <br />
            <span className="text-gradient">Nothing You Don't</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            A complete attendance management system designed for modern educational institutions.
            Powerful, intuitive, and beautiful.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <FeatureCard key={feature.title} feature={feature} index={index} />
          ))}
        </div>
        
        {/* Bottom stats */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mt-20 p-8 rounded-3xl bg-gradient-to-r from-primary/10 via-purple-500/10 to-pink-500/10 border border-primary/20 backdrop-blur-sm"
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { icon: Activity, value: '99.9%', label: 'Uptime' },
              { icon: Zap, value: '<500ms', label: 'Response Time' },
              { icon: Shield, value: '256-bit', label: 'Encryption' },
              { icon: Users, value: '24/7', label: 'Support' },
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                className="text-center"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4 + index * 0.1 }}
              >
                <div className="flex justify-center mb-3">
                  <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center">
                    <stat.icon className="w-6 h-6 text-primary" />
                  </div>
                </div>
                <div className="text-2xl md:text-3xl font-bold font-cyber text-gradient mb-1">
                  {stat.value}
                </div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
