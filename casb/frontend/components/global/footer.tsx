import Icons from "@/components/global/icons"
import { cn } from "@/lib/utils"
import { Heart } from 'lucide-react'
import Link from 'next/link'

interface FooterProps {
    className?: string;
}

const Footer = ({ className }: FooterProps) => {
    return (
        <footer className={cn("w-full border-t border-border", className)}>
            <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
                <div className="grid gap-8 lg:grid-cols-6">
                    {/* Brand section */}
                    <div className="lg:col-span-2">
                        <div className="flex items-center space-x-2">
                            {/* <Icons.Brain className="h-6 w-6" /> */}
                            <span className="text-lg font-semibold">ProctorAI</span>
                        </div>
                        <p className="mt-4 text-sm text-muted-foreground">
                            Secure and reliable AI-powered online examination proctoring platform for educational institutions
                        </p>
                    </div>

                    {/* Links sections */}
                    <div className="col-span-4 grid grid-cols-2 gap-8 sm:grid-cols-4">
                        {/* Product */}
                        <div>
                            <h3 className="text-sm font-semibold">Product</h3>
                            <ul className="mt-4 space-y-3 text-sm">
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Browser Extension
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Admin Dashboard
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        AI Detection
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Pricing Plans
                                    </Link>
                                </li>
                            </ul>
                        </div>

                        {/* Solutions */}
                        <div>
                            <h3 className="text-sm font-semibold">Solutions</h3>
                            <ul className="mt-4 space-y-3 text-sm">
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Universities
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Schools
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Training Centers
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Enterprises
                                    </Link>
                                </li>
                            </ul>
                        </div>

                        {/* Resources */}
                        <div>
                            <h3 className="text-sm font-semibold">Resources</h3>
                            <ul className="mt-4 space-y-3 text-sm">
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Documentation
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        API Reference
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Help Center
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Security
                                    </Link>
                                </li>
                            </ul>
                        </div>

                        {/* Company */}
                        <div>
                            <h3 className="text-sm font-semibold">Legal</h3>
                            <ul className="mt-4 space-y-3 text-sm">
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Privacy Policy
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Terms of Service
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        Cookie Policy
                                    </Link>
                                </li>
                                <li>
                                    <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">
                                        GDPR
                                    </Link>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Copyright */}
                <div className="mt-12 border-t border-border/40 pt-8">
                    <p className="text-sm text-muted-foreground">
                        &copy; {new Date().getFullYear()} ProctorAI. All rights reserved.
                    </p>
                </div>
            </div>
        </footer>
    )
}

export default Footer
