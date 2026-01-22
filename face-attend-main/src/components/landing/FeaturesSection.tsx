import { motion } from 'framer-motion';
import { 
  Camera, 
  BarChart3, 
  Clock, 
  Users, 
  BookOpen, 
  Shield,
  Zap,
  CheckCircle
} from 'lucide-react';

const features = [
  {
    icon: Camera,
    title: 'Face Registration',
    description: 'Students register their face once, then use it for all future attendance check-ins.',
    color: 'bg-primary/10 text-primary',
  },
  {
    icon: Zap,
    title: 'Instant Recognition',
    description: 'AI-powered recognition marks attendance in seconds with high accuracy.',
    color: 'bg-accent/10 text-accent',
  },
  {
    icon: BarChart3,
    title: 'Analytics Dashboard',
    description: 'View attendance percentages, trends, and detailed reports in real-time.',
    color: 'bg-success/10 text-success',
  },
  {
    icon: Clock,
    title: 'Automatic Absence',
    description: 'System automatically marks students absent if they miss the class period.',
    color: 'bg-warning/10 text-warning',
  },
  {
    icon: Users,
    title: 'Role-Based Access',
    description: 'Students, teachers, and admins each have tailored dashboards and permissions.',
    color: 'bg-primary/10 text-primary',
  },
  {
    icon: BookOpen,
    title: 'Class Management',
    description: 'Organize students by majors and classes with easy administration tools.',
    color: 'bg-accent/10 text-accent',
  },
];

export function FeaturesSection() {
  return (
    <section className="py-24 px-4 bg-secondary/30">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold font-display mb-4">
            Everything You Need
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            A complete attendance management system designed for modern educational institutions.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="p-6 rounded-2xl bg-card border border-border hover:border-primary/20 transition-all hover:shadow-lg group"
            >
              <div className={`w-14 h-14 rounded-xl ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                <feature.icon className="w-7 h-7" />
              </div>
              <h3 className="font-semibold text-xl mb-2">{feature.title}</h3>
              <p className="text-muted-foreground">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
