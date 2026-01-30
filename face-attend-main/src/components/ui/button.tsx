import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-semibold ring-offset-background transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90 hover:shadow-lg hover:shadow-primary/25 hover:-translate-y-0.5 active:translate-y-0",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90 hover:shadow-lg hover:shadow-destructive/25",
        outline: "border-2 border-input bg-background hover:bg-accent hover:text-accent-foreground hover:border-primary/50 hover:shadow-md",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80 hover:shadow-md",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
        hero: "relative overflow-hidden bg-gradient-to-r from-primary via-purple-500 to-pink-500 text-white font-bold shadow-lg shadow-primary/30 hover:shadow-xl hover:shadow-primary/40 hover:-translate-y-1 hover:scale-[1.02] active:scale-100 before:absolute before:inset-0 before:bg-gradient-to-r before:from-white/20 before:to-transparent before:translate-x-[-100%] hover:before:translate-x-[100%] before:transition-transform before:duration-700",
        heroOutline: "relative border-2 border-primary/50 bg-transparent text-foreground font-bold hover:border-primary hover:bg-primary/5 hover:shadow-lg hover:shadow-primary/20 hover:-translate-y-1 backdrop-blur-sm",
        glow: "relative bg-primary text-primary-foreground shadow-[0_0_20px_hsl(226_70%_55%_/_0.4)] hover:shadow-[0_0_30px_hsl(226_70%_55%_/_0.6),0_0_60px_hsl(280_80%_60%_/_0.3)] hover:-translate-y-1 transition-all duration-300",
        neon: "relative bg-transparent border-2 border-cyan-400 text-cyan-400 shadow-[0_0_10px_hsl(195_100%_50%_/_0.5),inset_0_0_10px_hsl(195_100%_50%_/_0.1)] hover:shadow-[0_0_20px_hsl(195_100%_50%_/_0.8),0_0_40px_hsl(195_100%_50%_/_0.4),inset_0_0_20px_hsl(195_100%_50%_/_0.2)] hover:text-white hover:bg-cyan-400/20 font-cyber tracking-wider",
        cyber: "relative bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 text-white font-bold uppercase tracking-wider shadow-lg hover:shadow-xl hover:shadow-purple-500/30 hover:-translate-y-1 before:absolute before:inset-[2px] before:bg-background before:rounded-[10px] before:z-[-1] overflow-hidden",
        success: "bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg shadow-green-500/25 hover:shadow-xl hover:shadow-green-500/40 hover:-translate-y-1",
        accent: "bg-gradient-to-r from-orange-500 to-pink-500 text-white shadow-lg shadow-orange-500/25 hover:shadow-xl hover:shadow-pink-500/40 hover:-translate-y-1",
      },
      size: {
        default: "h-11 px-5 py-2",
        sm: "h-9 rounded-lg px-4 text-xs",
        lg: "h-12 rounded-xl px-8 text-base",
        xl: "h-14 rounded-2xl px-10 text-lg",
        icon: "h-11 w-11",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
