import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Scan, Camera, Clock, Users, Shield, Zap } from 'lucide-react';

export default function Landing() {
  return (
    <div className="min-h-screen bg-background">
      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
                <Scan className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="font-bold text-xl">FaceAttend</span>
            </div>
            <div className="flex items-center gap-4">
              <Link to="/login">
                <Button variant="ghost">Login</Button>
              </Link>
              <Link to="/signup">
                <Button>Get Started</Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              Smart Face Recognition
              <br />
              <span className="text-primary">Attendance System</span>
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
              Streamline attendance tracking with AI-powered facial recognition.
              Fast, accurate, and secure.
            </p>
            <div className="flex gap-4 justify-center">
              <Link to="/signup">
                <Button size="lg" className="px-8">
                  Get Started Free
                </Button>
              </Link>
              <Link to="/login">
                <Button size="lg" variant="outline" className="px-8">
                  Sign In
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 bg-muted/30">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            Why Choose FaceAttend?
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: Camera,
                title: 'Face Recognition',
                description: 'Advanced FaceNet AI model for accurate face detection and matching.',
              },
              {
                icon: Zap,
                title: 'Real-time Processing',
                description: 'Instant face detection with live camera feed and bounding boxes.',
              },
              {
                icon: Shield,
                title: 'Secure Storage',
                description: 'All face data stored securely in MongoDB with privacy protection.',
              },
              {
                icon: Clock,
                title: 'Attendance History',
                description: 'Complete attendance records with timestamps and match distances.',
              },
              {
                icon: Users,
                title: 'Multi-user Support',
                description: 'Register multiple faces and track attendance for everyone.',
              },
              {
                icon: Scan,
                title: 'Easy Registration',
                description: 'Simple 4-photo registration process with guided capture.',
              },
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="bg-card p-6 rounded-xl border shadow-sm"
              >
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Get Started?</h2>
          <p className="text-muted-foreground mb-8">
            Join now and experience the future of attendance tracking.
          </p>
          <Link to="/signup">
            <Button size="lg" className="px-12">
              Create Free Account
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Scan className="w-5 h-5 text-primary" />
            <span className="font-semibold">FaceAttend</span>
          </div>
          <p className="text-sm text-muted-foreground">
            © 2026 FaceAttend. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
