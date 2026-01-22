import { motion } from 'framer-motion';
import { GraduationCap, UserCog, Shield } from 'lucide-react';

const roles = [
  {
    icon: GraduationCap,
    title: 'Students',
    features: [
      'Register face for attendance',
      'Check in using facial recognition',
      'View personal attendance percentage',
      'Track overall attendance stats',
      'Register for classes and majors',
    ],
    gradient: 'gradient-primary',
  },
  {
    icon: UserCog,
    title: 'Teachers',
    features: [
      'View real-time class attendance',
      'Monitor student check-ins live',
      'Mark absences manually',
      'Manage class subjects',
      'Generate attendance reports',
    ],
    gradient: 'gradient-accent',
  },
  {
    icon: Shield,
    title: 'Administrators',
    features: [
      'Create majors and classes',
      'Assign teachers to classes',
      'Verify student registrations',
      'Manage all users and roles',
      'System-wide analytics',
    ],
    gradient: 'gradient-success',
  },
];

export function RolesSection() {
  return (
    <section className="py-24 px-4">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold font-display mb-4">
            Built for Everyone
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Tailored experiences for students, teachers, and administrators.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {roles.map((role, index) => (
            <motion.div
              key={role.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.15 }}
              className="relative group"
            >
              <div className="absolute inset-0 rounded-3xl bg-gradient-to-b from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative p-8 rounded-3xl bg-card border border-border hover:border-primary/30 transition-all">
                <div className={`w-16 h-16 rounded-2xl ${role.gradient} flex items-center justify-center mb-6`}>
                  <role.icon className="w-8 h-8 text-primary-foreground" />
                </div>
                <h3 className="text-2xl font-bold font-display mb-4">{role.title}</h3>
                <ul className="space-y-3">
                  {role.features.map((feature) => (
                    <li key={feature} className="flex items-start gap-3 text-muted-foreground">
                      <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
