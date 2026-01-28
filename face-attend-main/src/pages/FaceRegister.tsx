import { useState, useRef, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { DashboardLayout } from '@/components/dashboard/DashboardLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Camera, CheckCircle, RefreshCw, Loader2, X, User } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { faceAPI } from '@/lib/face-api';

const NUM_IMAGES = 4;

interface DetectedFace {
  box: [number, number, number, number]; // [x, y, width, height]
  confidence: number;
  keypoints?: Record<string, [number, number]>;
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

  // Draw face detection boxes on overlay canvas
  const drawFaceBoxes = useCallback((faces: DetectedFace[], videoWidth: number, videoHeight: number) => {
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

      // Draw rounded rectangle border (green for registration)
      ctx.strokeStyle = '#00ff00';
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

      // Draw label - just show "Ready to capture" with detection confidence
      const label = `Ready to capture (${(confidence * 100).toFixed(0)}%)`;
      ctx.font = 'bold 14px Arial';
      const textWidth = ctx.measureText(label).width;
      const labelHeight = 24;
      const labelX = x;
      const labelY = y - labelHeight - 4;

      // Background for label
      ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
      ctx.beginPath();
      ctx.roundRect(labelX, Math.max(0, labelY), textWidth + 16, labelHeight, 6);
      ctx.fill();

      // Label text
      ctx.fillStyle = '#000000';
      ctx.fillText(label, labelX + 8, Math.max(17, labelY + 17));

      // Draw keypoints if available
      if (face.keypoints) {
        ctx.fillStyle = '#00ffff';
        Object.values(face.keypoints).forEach(([px, py]) => {
          ctx.beginPath();
          ctx.arc(px, py, 3, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    });
  }, []);

  // Run face detection periodically
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

      // Call detection API (identify=false for registration - just detect, don't identify)
      const result = await faceAPI.detectFace(imageBase64, 0.90, false);

      if (result.faces && result.faces.length > 0) {
        setDetectedFaces(result.faces);
        drawFaceBoxes(result.faces, video.videoWidth, video.videoHeight);
      } else {
        setDetectedFaces([]);
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
  }, [stream]);

  const capturePhoto = useCallback(() => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        setCapturedImages(prev => [...prev, imageData]);
        
        // Auto-stop camera after capturing all images
        if (capturedImages.length + 1 >= NUM_IMAGES) {
          stopCamera();
          setCurrentStep(3);
        }
      }
    }
  }, [stopCamera, capturedImages.length]);

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

  // Step indicator component
  const StepIndicator = () => (
    <div className="flex items-center justify-center gap-4 mb-6">
      {[1, 2, 3].map((step) => (
        <div key={step} className="flex items-center">
          <div
            className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold transition-all ${
              currentStep >= step
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted text-muted-foreground'
            }`}
          >
            {currentStep > step ? <CheckCircle className="w-5 h-5" /> : step}
          </div>
          {step < 3 && (
            <div
              className={`w-12 h-1 mx-2 rounded ${
                currentStep > step ? 'bg-primary' : 'bg-muted'
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );

  return (
    <DashboardLayout>
      <div className="max-w-2xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold font-display">Face Registration</h1>
          <p className="text-muted-foreground mt-1">
            Register your face for quick attendance check-ins.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="w-5 h-5 text-primary" />
              {isRegistered ? 'Registration Complete' : `Step ${currentStep} of 3`}
            </CardTitle>
            <CardDescription>
              {currentStep === 1 && 'Enter your name to get started.'}
              {currentStep === 2 && `Capture ${NUM_IMAGES} photos of your face.`}
              {currentStep === 3 && (isRegistered ? 'Your face has been registered!' : 'Review and confirm your photos.')}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <StepIndicator />

            {isRegistered ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center py-12"
              >
                <div className="w-20 h-20 rounded-full bg-green-500 flex items-center justify-center mx-auto mb-4">
                  <CheckCircle className="w-10 h-10 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Face Registered!</h3>
                <p className="text-muted-foreground mb-4">
                  {personName}'s face has been registered successfully.
                </p>
                <Button onClick={resetAll} variant="outline">
                  Register Another Person
                </Button>
              </motion.div>
            ) : (
              <>
                {/* Step 1: Enter Name */}
                {currentStep === 1 && (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="name">Your Name</Label>
                      <div className="relative">
                        <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                        <Input
                          id="name"
                          placeholder="Enter your full name"
                          value={personName}
                          onChange={(e) => setPersonName(e.target.value)}
                          className="pl-10"
                        />
                      </div>
                    </div>
                    <Button 
                      onClick={startCapture} 
                      className="w-full"
                      disabled={!personName.trim()}
                    >
                      <Camera className="w-4 h-4 mr-2" />
                      Continue to Photo Capture
                    </Button>
                  </div>
                )}

                {/* Step 2: Capture Photos */}
                {currentStep === 2 && (
                  <>
                    <div className="text-center mb-4">
                      <span className="text-2xl font-bold text-primary">
                        {capturedImages.length} / {NUM_IMAGES}
                      </span>
                      <span className="text-muted-foreground ml-2">photos captured</span>
                    </div>

                    <div className="relative aspect-video bg-black rounded-xl overflow-hidden" style={{ minHeight: '300px' }}>
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
                          <div className="text-center">
                            <Camera className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                            <p className="text-muted-foreground">Starting camera...</p>
                          </div>
                        </div>
                      )}
                    </div>

                    <canvas ref={canvasRef} className="hidden" />

                    {/* Detection status */}
                    {stream && (
                      <div className="flex items-center justify-center gap-2 text-sm">
                        <div className={`w-2 h-2 rounded-full ${detectedFaces.length > 0 ? 'bg-green-500' : 'bg-yellow-500'} ${isDetecting ? 'animate-pulse' : ''}`} />
                        <span className="text-muted-foreground">
                          {detectedFaces.length > 0
                            ? `Face detected (${(detectedFaces[0]?.confidence * 100).toFixed(0)}%)`
                            : 'Searching for face...'}
                        </span>
                      </div>
                    )}

                    {/* Captured images preview */}
                    {capturedImages.length > 0 && (
                      <div className="grid grid-cols-4 gap-2">
                        {capturedImages.map((img, index) => (
                          <div key={index} className="relative aspect-square rounded-lg overflow-hidden">
                            <img src={img} alt={`Capture ${index + 1}`} className="w-full h-full object-cover" />
                            <button
                              onClick={() => removeImage(index)}
                              className="absolute top-1 right-1 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center text-white hover:bg-red-600"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="flex gap-3">
                      {capturedImages.length < NUM_IMAGES && (
                        <Button
                          onClick={capturePhoto}
                          className="flex-1"
                          disabled={!stream || detectedFaces.length === 0}
                        >
                          <Camera className="w-4 h-4 mr-2" />
                          Capture Photo ({capturedImages.length + 1}/{NUM_IMAGES})
                        </Button>
                      )}
                      {capturedImages.length >= NUM_IMAGES && (
                        <Button
                          onClick={() => setCurrentStep(3)}
                          className="flex-1"
                        >
                          Review Photos
                        </Button>
                      )}
                    </div>
                  </>
                )}

                {/* Step 3: Review and Register */}
                {currentStep === 3 && !isRegistered && (
                  <>
                    <div className="text-center mb-4">
                      <p className="font-semibold">Registering: {personName}</p>
                      <p className="text-muted-foreground">{capturedImages.length} photos captured</p>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      {capturedImages.map((img, index) => (
                        <div key={index} className="aspect-square rounded-lg overflow-hidden border-2 border-primary/20">
                          <img src={img} alt={`Photo ${index + 1}`} className="w-full h-full object-cover" />
                        </div>
                      ))}
                    </div>

                    <div className="flex gap-3">
                      <Button
                        variant="outline"
                        className="flex-1"
                        onClick={() => {
                          setCapturedImages([]);
                          setCurrentStep(2);
                          startCamera();
                        }}
                      >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Retake All
                      </Button>
                      <Button
                        className="flex-1"
                        onClick={registerFace}
                        disabled={isRegistering}
                      >
                        {isRegistering ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Registering...
                          </>
                        ) : (
                          <>
                            <CheckCircle className="w-4 h-4 mr-2" />
                            Register Face
                          </>
                        )}
                      </Button>
                    </div>
                  </>
                )}
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Tips for Best Results</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                <span>Ensure good lighting on your face</span>
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                <span>Look directly at the camera</span>
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                <span>Remove glasses or hats if possible</span>
              </li>
              <li className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                <span>Vary your head position slightly between captures</span>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
