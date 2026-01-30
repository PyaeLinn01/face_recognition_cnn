import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { GraduationCap, UserCog, Shield, CheckCircle, ArrowRight } from 'lucide-react';
import { useRef } from 'react';

const roles = [
  {
    icon: GraduationCap,
    title: 'Students',
    subtitle: 'Check in with your face',
    features: [
      'Register face for attendance',
      'Check in using facial recognition',
      'View personal attendance percentage',
      'Track overall attendance stats',
      'Register for classes and majors',
    ],
    gradient: 'from-cyan-500 via-blue-500 to-purple-500',
    shadowColor: 'shadow-cyan-500/30',
    glowColor: 'cyan',
    bgPattern: 'radial-gradient(circle at 20% 80%, hsl(195 100% 50% / 0.1) 0%, transparent 50%)',
  },
  {
    icon: UserCog,
    title: 'Teachers',
    subtitle: 'Monitor in real-time',
    features: [
      'View real-time class attendance',
      'Monitor student check-ins live',
      'Mark absences manually',
      'Manage class subjects',
      'Generate attendance reports',
    ],
    gradient: 'from-orange-500 via-pink-500 to-purple-500',
    shadowColor: 'shadow-orange-500/30',
    glowColor: 'orange',
    bgPattern: 'radial-gradient(circle at 80% 20%, hsl(15 90% 60% / 0.1) 0%, transparent 50%)',
  },
  {
    icon: Shield,
    title: 'Administrators',
    subtitle: 'Full system control',
    features: [
      'Create majors and classes',
      'Assign teachers to classes',
      'Verify student registrations',
      'Manage all users and roles',
      'System-wide analytics',
    ],
    gradient: 'from-green-500 via-emerald-500 to-cyan-500',
    shadowColor: 'shadow-emerald-500/30',
    glowColor: 'emerald',
    bgPattern: 'radial-gradient(circle at 50% 50%, hsl(142 72% 45% / 0.1) 0%, transparent 50%)',
  },
];

// 3D Card component with tilt effect
const RoleCard = ({ role, index }: { role: typeof roles[0]; index: number }) => {
  const cardRef = useRef<HTMLDivElement>(null);
  
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  
  const mouseXSpring = useSpring(x, { stiffness: 150, damping: 20 });
  const mouseYSpring = useSpring(y, { stiffness: 150, damping: 20 });
  
  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ['15deg', '-15deg']);
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ['-15deg', '15deg']);
  
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const xPos = (e.clientX - rect.left) / rect.width - 0.5;
    const yPos = (e.clientY - rect.top) / rect.height - 0.5;
    x.set(xPos);
    y.set(yPos);
  };
  
  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.div
      ref={cardRef}
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.7, delay: index * 0.2 }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        perspective: 1000,
      }}
      className="relative group"
    >
      <motion.div
        style={{
          rotateX,
          rotateY,
          transformStyle: 'preserve-3d',
        }}
        className="relative"
      >
        {/* Gradient border glow */}
        <div className={`absolute -inset-[2px] rounded-[2rem] bg-gradient-to-r ${role.gradient} opacity-0 group-hover:opacity-70 blur-xl transition-opacity duration-500`} />
        <div className={`absolute -inset-[1px] rounded-[2rem] bg-gradient-to-r ${role.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
        
        {/* Card content */}
        <div 
          className="relative p-8 md:p-10 rounded-[2rem] bg-card border border-border backdrop-blur-xl overflow-hidden"
          style={{ background: role.bgPattern }}
        >
          {/* Animated background gradient */}
          <motion.div
            className={`absolute inset-0 bg-gradient-to-br ${role.gradient} opacity-0 group-hover:opacity-5 transition-opacity duration-500`}
          />
          
          {/* Floating orb decoration */}
          <motion.div
            className={`absolute -top-10 -right-10 w-40 h-40 rounded-full bg-gradient-to-br ${role.gradient} opacity-20 blur-2xl`}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.2, 0.3, 0.2],
            }}
            transition={{ duration: 4, repeat: Infinity }}
          />
          
          {/* Icon with 3D effect */}
          <motion.div
            className={`relative w-20 h-20 rounded-2xl bg-gradient-to-br ${role.gradient} ${role.shadowColor} shadow-xl flex items-center justify-center mb-8`}
            style={{ transform: 'translateZ(50px)' }}
            whileHover={{ scale: 1.1, rotate: 5 }}
          >
            <role.icon className="w-10 h-10 text-white" />
            
            {/* Sparkle effects */}
            {[...Array(4)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-1 h-1 rounded-full bg-white"
                style={{
                  top: `${20 + i * 20}%`,
                  left: `${20 + i * 15}%`,
                }}
                animate={{
                  opacity: [0, 1, 0],
                  scale: [0, 1.5, 0],
                }}
                transition={{
                  duration: 2,
                  delay: i * 0.3,
                  repeat: Infinity,
                }}
              />
            ))}
          </motion.div>
          
          {/* Title and subtitle */}
          <motion.div style={{ transform: 'translateZ(30px)' }}>
            <h3 className="text-3xl font-bold font-display mb-2">
              <span className={`bg-gradient-to-r ${role.gradient} bg-clip-text text-transparent`}>
                {role.title}
              </span>
            </h3>
            <p className="text-muted-foreground font-medium mb-6">{role.subtitle}</p>
          </motion.div>
          
          {/* Features list */}
          <ul className="space-y-4" style={{ transform: 'translateZ(20px)' }}>
            {role.features.map((feature, featureIndex) => (
              <motion.li
                key={feature}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.5 + featureIndex * 0.1 }}
                className="flex items-start gap-3 text-muted-foreground group/item"
              >
                <motion.div
                  className={`mt-0.5 w-5 h-5 rounded-full bg-gradient-to-r ${role.gradient} flex items-center justify-center flex-shrink-0`}
                  whileHover={{ scale: 1.2 }}
                >
                  <CheckCircle className="w-3 h-3 text-white" />
                </motion.div>
                <span className="group-hover/item:text-foreground transition-colors">{feature}</span>
              </motion.li>
            ))}
          </ul>
          
          {/* Learn more button */}
          <motion.button
            className={`mt-8 flex items-center gap-2 text-sm font-semibold bg-gradient-to-r ${role.gradient} bg-clip-text text-transparent group/btn`}
            whileHover={{ x: 5 }}
            style={{ transform: 'translateZ(25px)' }}
          >
            Learn more
            <motion.span
              animate={{ x: [0, 5, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              <ArrowRight className="w-4 h-4 text-primary" />
            </motion.span>
          </motion.button>
        </div>
      </motion.div>
    </motion.div>
  );
};

export function RolesSection() {
  return (
    <section className="relative py-32 px-4 overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute inset-0 gradient-mesh opacity-50" />
        
        {/* Animated orbs */}
        <motion.div
          className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-primary/10 blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, -50, 0],
          }}
          transition={{ duration: 20, repeat: Infinity }}
        />
        <motion.div
          className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full bg-accent/10 blur-3xl"
          animate={{
            x: [0, -80, 0],
            y: [0, 60, 0],
          }}
          transition={{ duration: 15, repeat: Infinity }}
        />
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
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-primary/10 via-purple-500/10 to-pink-500/10 border border-primary/20 text-sm font-semibold mb-6"
          >
            <span className="text-gradient">👥 Built for Everyone</span>
          </motion.div>
          
          <h2 className="text-4xl md:text-6xl lg:text-7xl font-bold font-display mb-6">
            One Platform,
            <br />
            <span className="text-gradient-cyber">Three Experiences</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Tailored dashboards and features for every user type.
            Everyone gets exactly what they need.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {roles.map((role, index) => (
            <RoleCard key={role.title} role={role} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
}
