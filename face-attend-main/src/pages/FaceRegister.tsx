import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Camera, CheckCircle, RefreshCw, Loader2, X, User, Sparkles, Lightbulb, Shield, Scan, Eye, EyeOff, AlertTriangle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { faceAPI } from '@/lib/face-api';

const NUM_IMAGES = 4;

interface DetectedFace {
  box: [number, number, number, number]; // [x, y, width, height]
  confidence: number;
  keypoints?: Record<string, [number, number]>;
}

interface LivenessResult {
  is_real: boolean;
  score: number;
  label: 'Real' | 'Fake' | 'No Face';
  face_detected: boolean;
}

export default function FaceRegister() {
  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const [personName, setPersonName] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [isRegistered, setIsRegistered] = useState(false);
  const [currentStep, setCurrentStep] = useState(1); // 1: Enter name, 2: Capture, 3: Review
  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Liveness detection state
  const [livenessResult, setLivenessResult] = useState<LivenessResult | null>(null);

  // Draw face detection boxes on overlay canvas with liveness status
  const drawFaceBoxes = useCallback((faces: DetectedFace[], videoWidth: number, videoHeight: number, liveness?: LivenessResult | null) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas size to video
    canvas.width = videoWidth;
    canvas.height = videoHeight;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach((face) => {
      const [x, y, w, h] = face.box;
      const confidence = face.confidence;
      
      // Determine colors based on liveness
      const isLive = liveness?.is_real === true;
      const isFake = liveness?.is_real === false && liveness?.face_detected;
      
      let boxColor: string;
      let labelBgColor: string;
      
      if (isFake) {
        boxColor = '#ff0000';
        labelBgColor = 'rgba(255, 0, 0, 0.9)';
      } else if (isLive) {
        boxColor = '#00ff00';
        labelBgColor = 'rgba(0, 255, 0, 0.9)';
      } else {
        boxColor = '#ffcc00';
        labelBgColor = 'rgba(255, 204, 0, 0.9)';
      }

      // Draw rounded rectangle border with glow
      ctx.save();
      if (isLive) {
        ctx.shadowColor = '#00ff00';
        ctx.shadowBlur = 15;
      } else if (isFake) {
        ctx.shadowColor = '#ff0000';
        ctx.shadowBlur = 20;
      }
      
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 3;
      ctx.beginPath();
      const radius = 12;
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + w - radius, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
      ctx.lineTo(x + w, y + h - radius);
      ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
      ctx.lineTo(x + radius, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();

      // Draw label with liveness status
      let label: string;
      if (isFake) {
        label = `⚠️ SPOOF - Cannot Register`;
      } else if (isLive) {
        label = `✓ LIVE - Ready (${(confidence * 100).toFixed(0)}%)`;
      } else {
        label = `Checking... (${(confidence * 100).toFixed(0)}%)`;
      }
      
      ctx.font = 'bold 14px Arial';
      const textWidth = ctx.measureText(label).width;
      const labelHeight = 24;
      const labelX = x;
      const labelY = y - labelHeight - 4;

      // Background for label
      ctx.fillStyle = labelBgColor;
      ctx.beginPath();
      ctx.roundRect(labelX, Math.max(0, labelY), textWidth + 16, labelHeight, 6);
      ctx.fill();

      // Label text
      ctx.fillStyle = isFake ? '#ffffff' : '#000000';
      ctx.fillText(label, labelX + 8, Math.max(17, labelY + 17));

      // Draw keypoints if available
      if (face.keypoints) {
        ctx.fillStyle = isLive ? '#00ffff' : (isFake ? '#ff6666' : '#ffff00');
        Object.values(face.keypoints).forEach(([px, py]) => {
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    });
  }, []);

  // Run face detection periodically with liveness check
  const runFaceDetection = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Check if video is ready
    if (video.readyState < 2 || video.videoWidth === 0) return;

    try {
      setIsDetecting(true);

      // Capture current frame
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0);
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.7);
      const imageBase64 = imageDataUrl.split(',')[1];

      // Call detection API and liveness API in parallel
      const [detectionResult, livenessResultData] = await Promise.all([
        faceAPI.detectFace(imageBase64, 0.90, false), // identify=false for registration
        faceAPI.checkLiveness(imageBase64).catch(() => null)
      ]);

      if (livenessResultData) {
        setLivenessResult(livenessResultData);
      }

      if (detectionResult.faces && detectionResult.faces.length > 0) {
        setDetectedFaces(detectionResult.faces);
        drawFaceBoxes(detectionResult.faces, video.videoWidth, video.videoHeight, livenessResultData);
      } else {
        setDetectedFaces([]);
        setLivenessResult(null);
        // Clear overlay
        const overlayCanvas = overlayCanvasRef.current;
        if (overlayCanvas) {
          const overlayCtx = overlayCanvas.getContext('2d');
          if (overlayCtx) {
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
          }
        }
      }
    } catch (error) {
      // Silently fail detection - don't show errors for each frame
      console.debug('Detection error:', error);
    } finally {
      setIsDetecting(false);
    }
  }, [drawFaceBoxes]);

  // Attach stream to video element when stream changes
  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(console.error);

      // Start face detection loop (every 500ms)
      detectionIntervalRef.current = setInterval(() => {
        runFaceDetection();
      }, 500);
    }
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream, runFaceDetection]);

  const startCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 }
      });
      setStream(mediaStream);
    } catch (error) {
      toast({
        variant: 'destructive',
        title: 'Camera Error',
        description: 'Could not access camera. Please check permissions.',
      });
    }
  }, [toast]);

  const stopCamera = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setDetectedFaces([]);
    setLivenessResult(null);
  }, [stream]);

  const capturePhoto = useCallback(async () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        const imageBase64 = imageData.split(',')[1];
        
        // Check liveness before accepting the capture
        try {
          const livenessCheck = await faceAPI.checkLiveness(imageBase64);
          
          if (!livenessCheck.face_detected) {
            toast({
              variant: 'destructive',
              title: 'No Face Detected',
              description: 'Please ensure your face is clearly visible.',
            });
            return;
          }
          
          if (!livenessCheck.is_real) {
            toast({
              variant: 'destructive',
              title: '⚠️ Spoof Detected!',
              description: `Cannot register a fake face. Score: ${livenessCheck.score.toFixed(2)}. Use your real face, not a photo or screen.`,
            });
            return;
          }
          
          // Liveness passed - accept the capture
          setCapturedImages(prev => [...prev, imageData]);
          
          toast({
            title: `✓ Photo ${capturedImages.length + 1} captured`,
            description: 'Live face verified successfully.',
          });
          
          // Auto-stop camera after capturing all images
          if (capturedImages.length + 1 >= NUM_IMAGES) {
            stopCamera();
            setCurrentStep(3);
          }
        } catch (error) {
          console.error('Liveness check error:', error);
          toast({
            variant: 'destructive',
            title: 'Verification Failed',
            description: 'Could not verify liveness. Please try again.',
          });
        }
      }
    }
  }, [stopCamera, capturedImages.length, toast]);

  const removeImage = (index: number) => {
    setCapturedImages(prev => prev.filter((_, i) => i !== index));
  };

  const resetAll = useCallback(() => {
    setCapturedImages([]);
    setIsRegistered(false);
    setCurrentStep(1);
    setPersonName('');
    stopCamera();
  }, [stopCamera]);

  const startCapture = () => {
    if (!personName.trim()) {
      toast({
        variant: 'destructive',
        title: 'Name Required',
        description: 'Please enter your name before capturing photos.',
      });
      return;
    }
    setCurrentStep(2);
    startCamera();
  };

  const registerFace = async () => {
    if (capturedImages.length < NUM_IMAGES) {
      toast({
        variant: 'destructive',
        title: 'Not Enough Photos',
        description: `Please capture ${NUM_IMAGES} photos.`,
      });
      return;
    }

    setIsRegistering(true);
    try {
      // Send each image to the backend
      for (let i = 0; i < capturedImages.length; i++) {
        const imageBase64 = capturedImages[i].split(',')[1]; // Remove data:image/jpeg;base64, prefix
        await faceAPI.registerFace(personName.trim(), imageBase64, i + 1);
      }

      setIsRegistered(true);
      setCurrentStep(3);
      toast({
        title: 'Face Registered!',
        description: `${personName}'s face has been registered successfully with ${NUM_IMAGES} images.`,
      });
    } catch (error: any) {
      console.error('Registration error:', error);
      toast({
        variant: 'destructive',
        title: 'Registration Failed',
        description: error?.response?.data?.error || 'Could not register your face. Please try again.',
      });
    } finally {
      setIsRegistering(false);
    }
  };

  // Step indicator component with enhanced animation
  const StepIndicator = () => (
    <div className="flex items-center justify-center gap-2 mb-8">
      {[
        { step: 1, icon: User, label: 'Identity' },
        { step: 2, icon: Camera, label: 'Capture' },
        { step: 3, icon: Shield, label: 'Register' }
      ].map(({ step, icon: Icon, label }, index) => (
        <div key={step} className="flex items-center">
          <motion.div 
            className="flex flex-col items-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <motion.div
              className={`relative w-14 h-14 rounded-2xl flex items-center justify-center text-sm font-semibold transition-all duration-500 ${
                currentStep >= step
                  ? 'bg-gradient-to-br from-primary to-violet-600 text-white shadow-lg shadow-primary/30'
                  : 'bg-muted/50 text-muted-foreground border border-border'
              }`}
              animate={currentStep === step ? { scale: [1, 1.05, 1] } : {}}
              transition={{ repeat: Infinity, duration: 2 }}
            >
              {currentStep > step ? (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring' }}
                >
                  <CheckCircle className="w-6 h-6" />
                </motion.div>
              ) : (
                <Icon className="w-6 h-6" />
              )}
              {currentStep === step && (
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-primary to-violet-600 animate-ping opacity-20" />
              )}
            </motion.div>
            <span className={`text-xs mt-2 font-medium ${currentStep >= step ? 'text-primary' : 'text-muted-foreground'}`}>
              {label}
            </span>
          </motion.div>
          {step < 3 && (
            <div className="relative w-16 mx-2 mb-6">
              <div className="absolute inset-0 h-1 bg-muted rounded" />
              <motion.div
                className="absolute inset-y-0 left-0 h-1 bg-gradient-to-r from-primary to-violet-600 rounded"
                initial={{ width: '0%' }}
                animate={{ width: currentStep > step ? '100%' : '0%' }}
                transition={{ duration: 0.5 }}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  );

  return (
    <DashboardLayout>
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Animated Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative"
        >
          <div className="absolute -top-4 -left-4 w-32 h-32 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-full blur-3xl" />
          <div className="absolute -top-2 right-0 w-24 h-24 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-full blur-3xl" />
          <h1 className="text-4xl font-bold font-display relative">
            <span className="text-gradient-primary">Face Registration</span>
          </h1>
          <p className="text-muted-foreground mt-2 text-lg">
            Register your face for quick attendance check-ins
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="relative overflow-hidden border-border/50">
            {/* Animated top gradient */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-cyan-500 to-violet-500" />
            
            {/* Background effects */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-primary/5 to-transparent rounded-full blur-3xl" />
            
            <CardHeader className="relative">
              <CardTitle className="flex items-center gap-3">
                <motion.div 
                  className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shadow-lg"
                  animate={!isRegistered ? { rotate: [0, 5, -5, 0] } : {}}
                  transition={{ repeat: Infinity, duration: 3 }}
                >
                  {isRegistered ? (
                    <CheckCircle className="w-6 h-6 text-white" />
                  ) : (
                    <Scan className="w-6 h-6 text-white" />
                  )}
                </motion.div>
                <span className="text-xl">
                  {isRegistered ? 'Registration Complete' : `Step ${currentStep} of 3`}
                </span>
              </CardTitle>
              <CardDescription className="text-base">
                {currentStep === 1 && 'Enter your name to get started.'}
                {currentStep === 2 && `Capture ${NUM_IMAGES} photos of your face.`}
                {currentStep === 3 && (isRegistered ? 'Your face has been registered!' : 'Review and confirm your photos.')}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6 relative">
              <StepIndicator />

            {isRegistered ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center py-12 relative"
              >
                {/* Success particles */}
                {[...Array(8)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute w-2 h-2 rounded-full bg-gradient-to-r from-emerald-400 to-green-500"
                    style={{ left: '50%', top: '50%' }}
                    initial={{ x: 0, y: 0, opacity: 1, scale: 1 }}
                    animate={{
                      x: Math.cos(i * 45 * Math.PI / 180) * 80,
                      y: Math.sin(i * 45 * Math.PI / 180) * 80,
                      opacity: 0,
                      scale: 0
                    }}
                    transition={{ duration: 1, delay: 0.2, ease: 'easeOut' }}
                  />
                ))}
                
                <motion.div 
                  className="relative w-24 h-24 mx-auto mb-6"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 200, delay: 0.1 }}
                >
                  <div className="absolute inset-0 rounded-full bg-gradient-to-br from-emerald-500 to-green-600 animate-pulse" />
                  <div className="relative w-full h-full rounded-full bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center shadow-lg shadow-emerald-500/30">
                    <motion.div
                      initial={{ scale: 0, rotate: -180 }}
                      animate={{ scale: 1, rotate: 0 }}
                      transition={{ type: 'spring', delay: 0.3 }}
                    >
                      <CheckCircle className="w-12 h-12 text-white" />
                    </motion.div>
                  </div>
                </motion.div>
                
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <h3 className="text-2xl font-bold font-display mb-2 flex items-center justify-center gap-2">
                    <Sparkles className="w-5 h-5 text-yellow-500" />
                    Face Registered!
                    <Sparkles className="w-5 h-5 text-yellow-500" />
                  </h3>
                  <p className="text-muted-foreground mb-6 text-lg">
                    <span className="font-semibold text-foreground">{personName}</span>'s face has been registered successfully.
                  </p>
                  <Button onClick={resetAll} variant="heroOutline" size="lg" className="group">
                    <RefreshCw className="w-4 h-4 mr-2 group-hover:rotate-180 transition-transform duration-500" />
                    Register Another Person
                  </Button>
                </motion.div>
              </motion.div>
            ) : (
              <>
                {/* Step 1: Enter Name */}
                {currentStep === 1 && (
                  <motion.div 
                    className="space-y-6"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                  >
                    <div className="space-y-3">
                      <Label htmlFor="name" className="text-base font-medium">Your Full Name</Label>
                      <div className="relative group">
                        <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/20 via-cyan-500/20 to-violet-500/20 blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity" />
                        <div className="relative">
                          <div className="absolute left-4 top-1/2 -translate-y-1/2 w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                            <User className="w-5 h-5 text-white" />
                          </div>
                          <Input
                            id="name"
                            placeholder="Enter your full name"
                            value={personName}
                            onChange={(e) => setPersonName(e.target.value)}
                            className="pl-20 h-14 text-lg rounded-xl border-border/50 focus:border-primary/50 transition-colors"
                          />
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground">This name will be associated with your face for attendance tracking</p>
                    </div>
                    <Button 
                      onClick={startCapture} 
                      className="w-full h-14 text-lg font-semibold"
                      variant="hero"
                      disabled={!personName.trim()}
                    >
                      <Camera className="w-5 h-5 mr-3" />
                      Continue to Photo Capture
                      <motion.span 
                        className="ml-2"
                        animate={{ x: [0, 5, 0] }}
                        transition={{ repeat: Infinity, duration: 1.5 }}
                      >
                        →
                      </motion.span>
                    </Button>
                  </motion.div>
                )}

                {/* Step 2: Capture Photos */}
                {currentStep === 2 && (
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                  >
                    {/* Progress indicator */}
                    <div className="text-center mb-6">
                      <div className="inline-flex items-center gap-4 px-6 py-3 rounded-2xl bg-gradient-to-r from-blue-500/10 via-cyan-500/10 to-violet-500/10 border border-primary/20">
                        <span className="text-3xl font-bold font-display text-gradient-primary">
                          {capturedImages.length}
                        </span>
                        <span className="text-2xl text-muted-foreground">/</span>
                        <span className="text-3xl font-bold text-muted-foreground">{NUM_IMAGES}</span>
                        <span className="text-muted-foreground font-medium">photos</span>
                      </div>
                    </div>

                    {/* Camera viewport */}
                    <div className="relative aspect-video bg-gradient-to-br from-gray-900 to-black rounded-2xl overflow-hidden shadow-2xl" style={{ minHeight: '320px' }}>
                      {/* Corner decorations */}
                      <div className="absolute top-4 left-4 w-12 h-12 border-l-2 border-t-2 border-cyan-400 rounded-tl-lg" />
                      <div className="absolute top-4 right-4 w-12 h-12 border-r-2 border-t-2 border-cyan-400 rounded-tr-lg" />
                      <div className="absolute bottom-4 left-4 w-12 h-12 border-l-2 border-b-2 border-cyan-400 rounded-bl-lg" />
                      <div className="absolute bottom-4 right-4 w-12 h-12 border-r-2 border-b-2 border-cyan-400 rounded-br-lg" />
                      
                      {/* Scan line animation */}
                      <motion.div 
                        className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent"
                        animate={{ top: ['0%', '100%', '0%'] }}
                        transition={{ repeat: Infinity, duration: 3, ease: 'linear' }}
                      />
                      
                      {stream ? (
                        <>
                          <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            className="w-full h-full object-cover"
                            style={{ display: 'block', width: '100%', height: '100%' }}
                          />
                          {/* Overlay canvas for face detection boxes */}
                          <canvas
                            ref={overlayCanvasRef}
                            className="absolute top-0 left-0 w-full h-full pointer-events-none"
                            style={{ objectFit: 'cover' }}
                          />
                        </>
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <motion.div 
                            className="text-center"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                          >
                            <motion.div
                              animate={{ rotate: 360 }}
                              transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
                              className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center mx-auto mb-4"
                            >
                              <Camera className="w-10 h-10 text-cyan-400" />
                            </motion.div>
                            <p className="text-cyan-400 font-medium">Initializing camera...</p>
                          </motion.div>
                        </div>
                      )}
                    </div>

                    <canvas ref={canvasRef} className="hidden" />

                    {/* Detection and Liveness status */}
                    {stream && (
                      <motion.div 
                        className="flex flex-col items-center gap-3 mt-4"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                      >
                        {/* Liveness Status */}
                        <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${
                          livenessResult?.is_real 
                            ? 'bg-emerald-500/10 border-emerald-500/30' 
                            : livenessResult?.face_detected && !livenessResult?.is_real
                              ? 'bg-red-500/10 border-red-500/30 animate-pulse' 
                              : 'bg-gray-500/10 border-gray-500/30'
                        }`}>
                          {livenessResult?.is_real ? (
                            <Eye className="w-4 h-4 text-emerald-400" />
                          ) : livenessResult?.face_detected ? (
                            <EyeOff className="w-4 h-4 text-red-400" />
                          ) : (
                            <Shield className="w-4 h-4 text-gray-400" />
                          )}
                          <span className={`text-sm font-medium ${
                            livenessResult?.is_real 
                              ? 'text-emerald-400' 
                              : livenessResult?.face_detected && !livenessResult?.is_real
                                ? 'text-red-400' 
                                : 'text-gray-400'
                          }`}>
                            {livenessResult?.is_real 
                              ? `✓ LIVE FACE - Ready to capture` 
                              : livenessResult?.face_detected && !livenessResult?.is_real
                                ? `⚠️ SPOOF DETECTED - Cannot capture`
                                : livenessResult && !livenessResult.face_detected
                                  ? 'No face for liveness check'
                                  : 'Waiting for face...'}
                          </span>
                        </div>

                        {/* Face Detection Status */}
                        <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-muted/50 border border-border">
                          <div className={`w-2.5 h-2.5 rounded-full ${detectedFaces.length > 0 ? 'bg-emerald-500' : 'bg-amber-500'} ${isDetecting ? 'animate-pulse' : ''}`} />
                          <span className="text-sm font-medium">
                            {detectedFaces.length > 0
                              ? `Face detected (${(detectedFaces[0]?.confidence * 100).toFixed(0)}% confidence)`
                              : 'Searching for face...'}
                          </span>
                        </div>
                      </motion.div>
                    )}

                    {/* Captured images preview */}
                    <AnimatePresence>
                      {capturedImages.length > 0 && (
                        <motion.div 
                          className="grid grid-cols-4 gap-3 mt-6"
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                        >
                          {capturedImages.map((img, index) => (
                            <motion.div 
                              key={index} 
                              className="relative aspect-square rounded-xl overflow-hidden border-2 border-emerald-500/50 shadow-lg group"
                              initial={{ scale: 0, opacity: 0 }}
                              animate={{ scale: 1, opacity: 1 }}
                              transition={{ type: 'spring', delay: index * 0.1 }}
                            >
                              <img src={img} alt={`Capture ${index + 1}`} className="w-full h-full object-cover" />
                              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                              <div className="absolute top-2 left-2 w-6 h-6 rounded-full bg-emerald-500 flex items-center justify-center text-white text-xs font-bold">
                                {index + 1}
                              </div>
                              <button
                                onClick={() => removeImage(index)}
                                className="absolute top-2 right-2 w-7 h-7 bg-red-500 rounded-full flex items-center justify-center text-white opacity-0 group-hover:opacity-100 transition-all hover:bg-red-600 hover:scale-110"
                              >
                                <X className="w-4 h-4" />
                              </button>
                            </motion.div>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>

                    <div className="flex gap-3 mt-6">
                      {capturedImages.length < NUM_IMAGES && (
                        <Button
                          onClick={capturePhoto}
                          className="flex-1 h-14 text-lg"
                          variant="hero"
                          disabled={!stream || detectedFaces.length === 0 || !livenessResult?.is_real}
                        >
                          <Camera className="w-5 h-5 mr-2" />
                          {livenessResult?.is_real 
                            ? `Capture (${capturedImages.length + 1}/${NUM_IMAGES})`
                            : livenessResult?.face_detected && !livenessResult?.is_real
                              ? 'Spoof Detected'
                              : detectedFaces.length === 0
                                ? 'No Face Detected'
                                : 'Verifying Liveness...'}
                        </Button>
                      )}
                      {capturedImages.length >= NUM_IMAGES && (
                        <Button
                          onClick={() => setCurrentStep(3)}
                          className="flex-1 h-14 text-lg"
                          variant="hero"
                        >
                          Review Photos
                          <motion.span 
                            className="ml-2"
                            animate={{ x: [0, 5, 0] }}
                            transition={{ repeat: Infinity, duration: 1.5 }}
                          >
                            →
                          </motion.span>
                        </Button>
                      )}
                    </div>
                  </motion.div>
                )}

                {/* Step 3: Review and Register */}
                {currentStep === 3 && !isRegistered && (
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                  >
                    <div className="text-center mb-6">
                      <div className="inline-flex items-center gap-3 px-6 py-3 rounded-2xl bg-gradient-to-r from-violet-500/10 via-purple-500/10 to-pink-500/10 border border-primary/20">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                          <User className="w-5 h-5 text-white" />
                        </div>
                        <div className="text-left">
                          <p className="text-sm text-muted-foreground">Registering</p>
                          <p className="font-bold text-lg">{personName}</p>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      {capturedImages.map((img, index) => (
                        <motion.div 
                          key={index} 
                          className="relative aspect-square rounded-2xl overflow-hidden group"
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: index * 0.1 }}
                          whileHover={{ scale: 1.02 }}
                        >
                          <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-primary/50 to-violet-500/50 p-0.5">
                            <div className="w-full h-full rounded-2xl overflow-hidden">
                              <img src={img} alt={`Photo ${index + 1}`} className="w-full h-full object-cover" />
                            </div>
                          </div>
                          <div className="absolute top-3 left-3 w-8 h-8 rounded-full bg-gradient-to-br from-primary to-violet-600 flex items-center justify-center text-white text-sm font-bold shadow-lg">
                            {index + 1}
                          </div>
                          <div className="absolute bottom-3 right-3">
                            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/90 text-white text-xs font-medium">
                              <CheckCircle className="w-3 h-3" />
                              Ready
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>

                    <div className="flex gap-4 mt-6">
                      <Button
                        variant="heroOutline"
                        className="flex-1 h-14 text-base"
                        onClick={() => {
                          setCapturedImages([]);
                          setCurrentStep(2);
                          startCamera();
                        }}
                      >
                        <RefreshCw className="w-5 h-5 mr-2" />
                        Retake All
                      </Button>
                      <Button
                        className="flex-1 h-14 text-base"
                        variant="hero"
                        onClick={registerFace}
                        disabled={isRegistering}
                      >
                        {isRegistering ? (
                          <>
                            <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                            Registering...
                          </>
                        ) : (
                          <>
                            <Shield className="w-5 h-5 mr-2" />
                            Register Face
                          </>
                        )}
                      </Button>
                    </div>
                  </motion.div>
                )}
              </>
            )}
          </CardContent>
        </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="relative overflow-hidden border-border/50 hover:border-amber-500/30 transition-colors">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-amber-500 via-yellow-500 to-orange-500" />
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
                  <Lightbulb className="w-5 h-5 text-white" />
                </div>
                Tips for Best Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="grid sm:grid-cols-2 gap-3">
                {[
                  { text: 'Ensure good lighting on your face', icon: '💡' },
                  { text: 'Look directly at the camera', icon: '👀' },
                  { text: 'Remove glasses or hats if possible', icon: '🎩' },
                  { text: 'Vary head position slightly between captures', icon: '🔄' }
                ].map((tip, i) => (
                  <motion.li 
                    key={i}
                    className="flex items-start gap-3 p-3 rounded-xl bg-muted/50 hover:bg-muted transition-colors"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 + i * 0.1 }}
                  >
                    <span className="text-xl">{tip.icon}</span>
                    <span className="text-sm text-muted-foreground">{tip.text}</span>
                  </motion.li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </DashboardLayout>
  );
}
