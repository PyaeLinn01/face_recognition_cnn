import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Camera, CheckCircle, Loader2, XCircle, RefreshCw, Scan, Sparkles, Shield, AlertTriangle, Lightbulb, Eye, EyeOff } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { faceAPI } from '@/lib/face-api';

interface DetectedFace {
  box: [number, number, number, number]; // [x, y, width, height]
  confidence: number;
  keypoints?: Record<string, [number, number]>;
  identity?: string | null;  // Identified person name
  match_confidence?: number;  // Match confidence (1 - distance)
}

interface VerificationResult {
  matched: boolean;
  identity: string;
  distance: number;
  threshold: number;
  face_detected?: boolean;
  face_confidence?: number;
}

interface LivenessResult {
  is_real: boolean;
  score: number;
  label: 'Real' | 'Fake' | 'No Face';
  face_detected: boolean;
}

export default function MarkAttendance() {
  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [isMarked, setIsMarked] = useState(false);
  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Liveness detection state
  const [livenessResult, setLivenessResult] = useState<LivenessResult | null>(null);
  const [isCheckingLiveness, setIsCheckingLiveness] = useState(false);
  const livenessIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Draw face detection boxes on overlay canvas
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
      const identity = face.identity;
      const matchConfidence = face.match_confidence || 0;

      // Determine colors based on liveness and identity
      const isIdentified = identity && matchConfidence > 0;
      const isLive = liveness?.is_real === true;
      const isFake = liveness?.is_real === false && liveness?.face_detected;
      
      let boxColor: string;
      let labelBgColor: string;
      
      if (isFake) {
        // Red for fake/spoof
        boxColor = '#ff0000';
        labelBgColor = 'rgba(255, 0, 0, 0.9)';
      } else if (isLive && isIdentified) {
        // Green for live + identified
        boxColor = '#00ff00';
        labelBgColor = 'rgba(0, 255, 0, 0.9)';
      } else if (isLive) {
        // Cyan for live but not identified
        boxColor = '#00ffcc';
        labelBgColor = 'rgba(0, 255, 204, 0.9)';
      } else {
        // Yellow for unknown liveness
        boxColor = '#ffcc00';
        labelBgColor = 'rgba(255, 204, 0, 0.9)';
      }

      // Draw rounded rectangle border with glow effect for liveness
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

      // Create label text
      let label: string;
      if (isFake) {
        label = `⚠️ FAKE FACE DETECTED (${(liveness?.score || 0).toFixed(2)})`;
      } else if (isIdentified) {
        const distance = 1 - matchConfidence;
        const livenessText = isLive ? '✓ LIVE' : '';
        label = `${livenessText} ${identity} (dist: ${distance.toFixed(3)})`;
      } else {
        label = isLive ? '✓ Live - Unknown Face' : 'Unknown Face';
      }

      ctx.font = 'bold 14px Arial';
      const textWidth = ctx.measureText(label).width;
      const labelHeight = 28;
      const labelX = x;
      const labelY = y - labelHeight - 4;

      // Background for label
      ctx.fillStyle = labelBgColor;
      ctx.beginPath();
      ctx.roundRect(labelX, Math.max(0, labelY), textWidth + 16, labelHeight, 6);
      ctx.fill();

      // Label text
      ctx.fillStyle = isFake ? '#ffffff' : '#000000';
      ctx.fillText(label, labelX + 8, Math.max(17, labelY + 19));

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

  // Run liveness detection
  const runLivenessCheck = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isVerifying) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.readyState < 2 || video.videoWidth === 0) return;

    try {
      setIsCheckingLiveness(true);

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0);
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.7);
      const imageBase64 = imageDataUrl.split(',')[1];

      // Call liveness API
      const result = await faceAPI.checkLiveness(imageBase64);
      setLivenessResult(result);
    } catch (error) {
      console.debug('Liveness check error:', error);
    } finally {
      setIsCheckingLiveness(false);
    }
  }, [isVerifying]);

  // Run face detection periodically
  const runFaceDetection = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isVerifying) return;

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
        faceAPI.detectFace(imageBase64, 0.90),
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
  }, [isVerifying, drawFaceBoxes]);

  // Start detection loop when stream is active
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

  const verifyAndMark = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    // Stop detection during verification
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }

    setIsVerifying(true);

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Could not get canvas context');

      ctx.drawImage(video, 0, 0);
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      const imageBase64 = imageDataUrl.split(',')[1];

      // First check liveness
      const livenessCheck = await faceAPI.checkLiveness(imageBase64);
      
      if (!livenessCheck.face_detected) {
        toast({
          variant: 'destructive',
          title: 'No Face Detected',
          description: 'Please ensure your face is clearly visible in the camera.',
        });
        // Restart detection
        detectionIntervalRef.current = setInterval(() => {
          runFaceDetection();
        }, 500);
        setIsVerifying(false);
        return;
      }
      
      if (!livenessCheck.is_real) {
        toast({
          variant: 'destructive',
          title: '⚠️ Spoof Detected!',
          description: `Anti-spoofing check failed. Score: ${livenessCheck.score.toFixed(2)}. Please use a real face, not a photo or screen.`,
        });
        // Restart detection
        detectionIntervalRef.current = setInterval(() => {
          runFaceDetection();
        }, 500);
        setIsVerifying(false);
        return;
      }

      // Liveness passed - now verify identity
      const verifyResult = await faceAPI.verifyFace(imageBase64, 0.6, true);

      setResult({
        matched: verifyResult.matched,
        identity: verifyResult.identity || 'Unknown',
        distance: verifyResult.distance || 999,
        threshold: verifyResult.threshold || 0.6,
        face_detected: verifyResult.face_detected,
        face_confidence: verifyResult.face_confidence,
      });

      if (verifyResult.matched) {
        await faceAPI.recordAttendance(
          verifyResult.identity,
          verifyResult.identity,
          verifyResult.distance
        );

        setIsMarked(true);
        stopCamera();
        toast({
          title: '✓ Attendance Marked!',
          description: `Welcome, ${verifyResult.identity}! Liveness verified. Your attendance has been recorded.`,
        });
      } else {
        const message = verifyResult.face_detected
          ? `Live face detected but not recognized. Distance: ${verifyResult.distance?.toFixed(3) || 'N/A'}`
          : 'No face detected. Please ensure your face is visible and well-lit.';
        toast({
          variant: 'destructive',
          title: 'Verification Failed',
          description: message,
        });
        // Restart detection
        detectionIntervalRef.current = setInterval(() => {
          runFaceDetection();
        }, 500);
      }
    } catch (error: any) {
      console.error('Verification error:', error);
      toast({
        variant: 'destructive',
        title: 'Verification Failed',
        description: error?.response?.data?.error || 'Could not verify face. Please try again.',
      });
      // Restart detection
      detectionIntervalRef.current = setInterval(() => {
        runFaceDetection();
      }, 500);
    } finally {
      setIsVerifying(false);
    }
  }, [stopCamera, toast, runFaceDetection]);

  // Reset function should also clear liveness
  const handleReset = useCallback(() => {
    setResult(null);
    setIsMarked(false);
    setDetectedFaces([]);
    setLivenessResult(null);
    startCamera();
  }, [startCamera]);

  return (
    <DashboardLayout>
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Animated Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative"
        >
          <div className="absolute -top-4 -left-4 w-32 h-32 bg-gradient-to-br from-emerald-500/20 to-green-500/20 rounded-full blur-3xl" />
          <div className="absolute -top-2 right-0 w-24 h-24 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-full blur-3xl" />
          <h1 className="text-4xl font-bold font-display relative">
            <span className="text-gradient-primary">Mark Attendance</span>
          </h1>
          <p className="text-muted-foreground mt-2 text-lg">
            Use facial recognition to mark your attendance
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="relative overflow-hidden border-border/50">
            {/* Animated top gradient */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 via-green-500 to-cyan-500" />
            
            {/* Background effects */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-emerald-500/5 to-transparent rounded-full blur-3xl" />
            
            <CardHeader className="relative">
              <CardTitle className="flex items-center gap-3">
                <motion.div 
                  className="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center shadow-lg"
                  animate={!isMarked ? { rotate: [0, 5, -5, 0] } : {}}
                  transition={{ repeat: Infinity, duration: 3 }}
                >
                  {isMarked ? (
                    <CheckCircle className="w-6 h-6 text-white" />
                  ) : (
                    <Scan className="w-6 h-6 text-white" />
                  )}
                </motion.div>
                <span className="text-xl">Face Verification</span>
              </CardTitle>
              <CardDescription className="text-base">
                Look at the camera to verify your identity
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6 relative">
              <AnimatePresence mode="wait">
            {isMarked && result?.matched ? (
              <motion.div
                key="success"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="text-center py-12 relative"
              >
                {/* Success particles */}
                {[...Array(10)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute w-2 h-2 rounded-full bg-gradient-to-r from-emerald-400 to-green-500"
                    style={{ left: '50%', top: '40%' }}
                    initial={{ x: 0, y: 0, opacity: 1, scale: 1 }}
                    animate={{
                      x: Math.cos(i * 36 * Math.PI / 180) * 100,
                      y: Math.sin(i * 36 * Math.PI / 180) * 100,
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
                    Attendance Marked!
                    <Sparkles className="w-5 h-5 text-yellow-500" />
                  </h3>
                  <p className="text-lg mb-2">
                    Welcome, <span className="font-bold text-gradient-primary">{result.identity}</span>
                  </p>
                  <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 mb-6">
                    <Shield className="w-4 h-4 text-emerald-500" />
                    <span className="text-sm">
                      Match Distance: <span className="font-mono font-bold text-emerald-500">{result.distance.toFixed(4)}</span>
                    </span>
                  </div>
                  <div className="block">
                    <Button onClick={handleReset} variant="heroOutline" size="lg" className="group">
                      <RefreshCw className="w-4 h-4 mr-2 group-hover:rotate-180 transition-transform duration-500" />
                      Mark Another
                    </Button>
                  </div>
                </motion.div>
              </motion.div>
            ) : result && !result.matched ? (
              <motion.div
                key="failure"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="text-center py-12"
              >
                <motion.div 
                  className="relative w-24 h-24 mx-auto mb-6"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 200 }}
                >
                  <div className="absolute inset-0 rounded-full bg-gradient-to-br from-red-500 to-rose-600 animate-pulse" />
                  <div className="relative w-full h-full rounded-full bg-gradient-to-br from-red-500 to-rose-600 flex items-center justify-center shadow-lg shadow-red-500/30">
                    {result.face_detected ? (
                      <XCircle className="w-12 h-12 text-white" />
                    ) : (
                      <AlertTriangle className="w-12 h-12 text-white" />
                    )}
                  </div>
                </motion.div>
                
                <h3 className="text-2xl font-bold font-display mb-3">
                  {result.face_detected ? 'Face Not Recognized' : 'No Face Detected'}
                </h3>
                <p className="text-muted-foreground mb-4 max-w-sm mx-auto">
                  {result.face_detected
                    ? 'Your face was detected but not matched to any registered identity.'
                    : 'Please ensure your face is visible and well-lit.'}
                </p>
                {result.face_detected && (
                  <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-red-500/10 border border-red-500/20 mb-6">
                    <span className="text-sm text-red-400">
                      Distance: <span className="font-mono font-bold">{result.distance.toFixed(3)}</span>
                      <span className="mx-2 text-muted-foreground">|</span>
                      Threshold: <span className="font-mono">{result.threshold}</span>
                    </span>
                  </div>
                )}
                <div className="flex gap-3 justify-center">
                  <Button onClick={handleReset} variant="heroOutline" size="lg">
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Try Again
                  </Button>
                  <Button asChild variant="hero" size="lg">
                    <a href="/dashboard/face-register">Register Face</a>
                  </Button>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="camera"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {/* Video container with overlay canvas */}
                <div className="relative aspect-video bg-gradient-to-br from-gray-900 to-black rounded-2xl overflow-hidden shadow-2xl" style={{ minHeight: '320px' }}>
                  {/* Corner decorations */}
                  <div className="absolute top-4 left-4 w-12 h-12 border-l-2 border-t-2 border-emerald-400 rounded-tl-lg z-10" />
                  <div className="absolute top-4 right-4 w-12 h-12 border-r-2 border-t-2 border-emerald-400 rounded-tr-lg z-10" />
                  <div className="absolute bottom-4 left-4 w-12 h-12 border-l-2 border-b-2 border-emerald-400 rounded-bl-lg z-10" />
                  <div className="absolute bottom-4 right-4 w-12 h-12 border-r-2 border-b-2 border-emerald-400 rounded-br-lg z-10" />
                  
                  {/* Scan line animation */}
                  {stream && (
                    <motion.div 
                      className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-emerald-400 to-transparent z-10"
                      animate={{ top: ['0%', '100%', '0%'] }}
                      transition={{ repeat: Infinity, duration: 3, ease: 'linear' }}
                    />
                  )}
                  
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
                          className="w-20 h-20 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-green-500/20 flex items-center justify-center mx-auto mb-4"
                          animate={{ scale: [1, 1.05, 1] }}
                          transition={{ repeat: Infinity, duration: 2 }}
                        >
                          <Camera className="w-10 h-10 text-emerald-400" />
                        </motion.div>
                        <p className="text-emerald-400 font-medium">Camera not started</p>
                        <p className="text-sm text-muted-foreground mt-1">Click below to start</p>
                      </motion.div>
                    </div>
                  )}
                </div>

                {/* Hidden canvas for capture */}
                <canvas ref={canvasRef} className="hidden" />

                {/* Detection status */}
                {stream && (
                  <motion.div 
                    className="flex flex-col items-center gap-3 mt-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    {/* Liveness Status Indicator */}
                    <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${
                      livenessResult?.is_real 
                        ? 'bg-emerald-500/10 border-emerald-500/30' 
                        : livenessResult?.face_detected && !livenessResult?.is_real
                          ? 'bg-red-500/10 border-red-500/30 animate-pulse' 
                          : isCheckingLiveness
                            ? 'bg-blue-500/10 border-blue-500/30'
                            : 'bg-gray-500/10 border-gray-500/30'
                    }`}>
                      {livenessResult?.is_real ? (
                        <Eye className="w-4 h-4 text-emerald-400" />
                      ) : livenessResult?.face_detected ? (
                        <EyeOff className="w-4 h-4 text-red-400" />
                      ) : isCheckingLiveness ? (
                        <Shield className="w-4 h-4 text-blue-400 animate-spin" />
                      ) : (
                        <Shield className="w-4 h-4 text-gray-400" />
                      )}
                      <span className={`text-sm font-medium ${
                        livenessResult?.is_real 
                          ? 'text-emerald-400' 
                          : livenessResult?.face_detected && !livenessResult?.is_real
                            ? 'text-red-400' 
                            : isCheckingLiveness
                              ? 'text-blue-400'
                              : 'text-gray-400'
                      }`}>
                        {livenessResult?.is_real 
                          ? `✓ LIVE FACE (${(livenessResult.score * 100).toFixed(0)}%)` 
                          : livenessResult?.face_detected && !livenessResult?.is_real
                            ? `⚠️ SPOOF DETECTED (${(livenessResult.score * 100).toFixed(0)}%)`
                            : isCheckingLiveness
                              ? 'Checking liveness...'
                              : livenessResult && !livenessResult.face_detected
                                ? 'No face detected for liveness'
                                : 'Waiting for face...'}
                      </span>
                    </div>

                    {/* Face Detection Status */}
                    <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${
                      detectedFaces.length > 0 && detectedFaces[0]?.identity 
                        ? 'bg-emerald-500/10 border-emerald-500/30' 
                        : detectedFaces.length > 0 
                          ? 'bg-amber-500/10 border-amber-500/30' 
                          : 'bg-red-500/10 border-red-500/30'
                    }`}>
                      <div className={`w-2.5 h-2.5 rounded-full ${
                        detectedFaces.length > 0 && detectedFaces[0]?.identity 
                          ? 'bg-emerald-500' 
                          : detectedFaces.length > 0 
                            ? 'bg-amber-500' 
                            : 'bg-red-500'
                      } ${isDetecting ? 'animate-pulse' : ''}`} />
                      <span className={`text-sm font-medium ${
                        detectedFaces.length > 0 && detectedFaces[0]?.identity 
                          ? 'text-emerald-400' 
                          : detectedFaces.length > 0 
                            ? 'text-amber-400' 
                            : 'text-red-400'
                      }`}>
                        {detectedFaces.length > 0 && detectedFaces[0]?.identity
                          ? `Welcome, ${detectedFaces[0].identity}!`
                          : detectedFaces.length > 0
                            ? 'Face detected - Unknown'
                            : 'Searching for face...'}
                      </span>
                    </div>
                  </motion.div>
                )}

                <div className="flex gap-4 mt-6">
                  {!stream ? (
                    <Button className="flex-1 h-14 text-lg" variant="hero" onClick={startCamera}>
                      <Camera className="w-5 h-5 mr-2" />
                      Start Camera
                    </Button>
                  ) : (
                    <Button
                      className="flex-1 h-14 text-lg"
                      variant="hero"
                      onClick={verifyAndMark}
                      disabled={isVerifying || detectedFaces.length === 0}
                    >
                      {isVerifying ? (
                        <>
                          <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                          Verifying...
                        </>
                      ) : (
                        <>
                          <Shield className="w-5 h-5 mr-2" />
                          Verify & Mark Attendance
                        </>
                      )}
                    </Button>
                  )}
                </div>
              </motion.div>
            )}
              </AnimatePresence>
          </CardContent>
        </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="relative overflow-hidden border-border/50 hover:border-cyan-500/30 transition-colors">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-violet-500" />
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
                  <Lightbulb className="w-5 h-5 text-white" />
                </div>
                Instructions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="grid sm:grid-cols-2 gap-3">
                {[
                  { text: 'Make sure your face is registered first', icon: '📝' },
                  { text: 'Wait for the green box around your face', icon: '🟢' },
                  { text: 'Ensure good lighting on your face', icon: '💡' },
                  { text: 'Click verify when the box appears', icon: '✅' }
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
