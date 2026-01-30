import { Navbar } from '@/components/landing/Navbar';
import { HeroSection } from '@/components/landing/HeroSection';
import { FeaturesSection } from '@/components/landing/FeaturesSection';
import { RolesSection } from '@/components/landing/RolesSection';
import { Footer } from '@/components/landing/Footer';

export default function Landing() {
  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background effects */}
      <div className="fixed inset-0 gradient-mesh opacity-30 pointer-events-none" />
      <div className="fixed inset-0 cyber-grid opacity-20 pointer-events-none" />
      
      {/* Floating orbs */}
      <div className="fixed top-1/4 left-1/4 w-96 h-96 bg-gradient-to-br from-primary/20 to-violet-500/20 rounded-full blur-3xl animate-float pointer-events-none" />
      <div className="fixed bottom-1/4 right-1/4 w-80 h-80 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-full blur-3xl animate-float-slow pointer-events-none" />
      
      <Navbar />
      <HeroSection />
      <FeaturesSection />
      <RolesSection />
      <Footer />
    </div>
  );
}
