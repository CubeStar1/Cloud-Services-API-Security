'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import Image from "next/image"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { InfoIcon, Terminal, ArrowRight, Monitor } from "lucide-react"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export default function FlowchartPage() {
    return (
        <div className="flex-1 space-y-6 p-4 md:p-8 pt-6">
            <div>
                <h2 className="text-3xl font-bold tracking-tight">System Workflow</h2>
                <p className="text-muted-foreground">
                    Complete pipeline from data collection to model deployment
                </p>
            </div>

            <Tabs defaultValue="overview" className="space-y-4">
                <TabsList>
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="gui">GUI Usage</TabsTrigger>
                    <TabsTrigger value="terminal">Terminal Usage</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-4">
                    <div className="relative aspect-[16/9] w-full overflow-hidden rounded-lg border">
                        <Image
                            src="/flow-diagram.png"
                            alt="System workflow diagram"
                            fill
                            className="object-contain"
                            priority
                        />
                    </div>

                    <div className="grid gap-6">
                        <Card>
                            <CardHeader>
                                <CardTitle>1. Data Collection</CardTitle>
                                <CardDescription>Network traffic capture and processing</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <p>
                                    Using AnyProxy, the system captures network traffic and saves it as JSON files for each service.
                                    This raw data forms the foundation for our analysis pipeline.
                                </p>
                                <Alert>
                                    <InfoIcon className="h-4 w-4" />
                                    <AlertDescription>
                                        Located in <code>/data-collection/manual</code> using AnyProxy rules
                                    </AlertDescription>
                                </Alert>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>2. DeBERTa Zero-Shot Learning</CardTitle>
                                <CardDescription>Initial unsupervised labelling</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <p>
                                    The DeBERTa model performs initial unsupervised labelling of the traffic data.
                                    If confidence is below 0.5, the system requests more activity data for that service.
                                </p>
                                <Alert>
                                    <InfoIcon className="h-4 w-4" />
                                    <AlertDescription>
                                        Implemented in <code>/zsl/deberta/inference.py</code>
                                    </AlertDescription>
                                </Alert>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>3. CodeBERT Processing</CardTitle>
                                <CardDescription>Service and activity classification</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <p>
                                    When DeBERTa's confidence exceeds 0.5, the data is processed by CodeBERT
                                    for more detailed service and activity classification.
                                </p>
                                <Alert>
                                    <InfoIcon className="h-4 w-4" />
                                    <AlertDescription>
                                        Located in <code>/zsl/codebert/{'{train,inference}'}.py</code>
                                    </AlertDescription>
                                </Alert>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>4. Random Forest Training</CardTitle>
                                <CardDescription>Machine learning classification</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <p>
                                    Using scikit-learn and Python, a Random Forest classifier is trained on the
                                    labeled data from previous steps. This provides a robust classification model.
                                </p>
                                <Alert>
                                    <InfoIcon className="h-4 w-4" />
                                    <AlertDescription>
                                        Implemented in <code>/rfc/train.py</code>
                                    </AlertDescription>
                                </Alert>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>5. C Code Generation</CardTitle>
                                <CardDescription>Model deployment optimization</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <p>
                                    The trained Random Forest model is converted into optimized if-else decision
                                    trees in C, enabling efficient execution on live traffic data.
                                </p>
                                <Alert>
                                    <InfoIcon className="h-4 w-4" />
                                    <AlertDescription>
                                        Generated C code is used for real-time traffic classification
                                    </AlertDescription>
                                </Alert>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="gui" className="space-y-4">
                    <div className="grid gap-6">
                        <Card>
                            <CardHeader>
                                <CardTitle>Step 1: Data Collection</CardTitle>
                                <CardDescription>Setting up traffic capture</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <p>1. Start AnyProxy to capture traffic:</p>
                                    <Alert>
                                        <Terminal className="h-4 w-4" />
                                        <AlertDescription className="font-mono">
                                            anyproxy --port 8001 --rule general-json-key.js
                                        </AlertDescription>
                                    </Alert>
                                </div>
                                <div className="space-y-2">
                                    <p>2. View captured logs in the Raw Logs page</p>
                                    <Button variant="outline" asChild>
                                        <Link href="/logs">
                                            Go to Raw Logs <ArrowRight className="ml-2 h-4 w-4" />
                                        </Link>
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Step 2: Data Labelling</CardTitle>
                                <CardDescription>Process raw logs into labeled data</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <p>1. Convert JSON logs to CSV format using the Raw Logs page</p>
                                    <p>2. Access the Labelling page to start manual labelling:</p>
                                    <Button variant="outline" asChild>
                                        <Link href="/labelling">
                                            Go to Labelling <ArrowRight className="ml-2 h-4 w-4" />
                                        </Link>
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Step 3: DeBERTa Classification</CardTitle>
                                <CardDescription>Initial zero-shot classification</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <p>1. Access the DeBERTa page to start classification</p>
                                    <p>2. Click "Start Inference" to process the CSV files</p>
                                    <Button variant="outline" asChild>
                                        <Link href="/zsl/deberta">
                                            Go to DeBERTa <ArrowRight className="ml-2 h-4 w-4" />
                                        </Link>
                                    </Button>
                                </div>
                                <Alert>
                                    <InfoIcon className="h-4 w-4" />
                                    <AlertDescription>
                                        If confidence is below 0.5, collect more data for that service
                                    </AlertDescription>
                                </Alert>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Step 4: CodeBERT Processing</CardTitle>
                                <CardDescription>Advanced classification and training</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <p>1. Train the CodeBERT model using the training data</p>
                                    <p>2. Run inference on the processed data</p>
                                    <Button variant="outline" asChild>
                                        <Link href="/zsl/codebert">
                                            Go to CodeBERT <ArrowRight className="ml-2 h-4 w-4" />
                                        </Link>
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Step 5: Random Forest & Deployment</CardTitle>
                                <CardDescription>Final model training and optimization</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <p>1. Train the Random Forest model on the labeled data</p>
                                    <p>2. Generate optimized C code for deployment</p>
                                    <Button variant="outline" asChild>
                                        <Link href="/random-forest">
                                            Go to Random Forest <ArrowRight className="ml-2 h-4 w-4" />
                                        </Link>
                                    </Button>
                                </div>
                                <Alert>
                                    <InfoIcon className="h-4 w-4" />
                                    <AlertDescription>
                                        The generated C code can be used for real-time classification of live traffic
                                    </AlertDescription>
                                </Alert>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="terminal" className="space-y-4">
                    <div className="grid gap-6">
                        <Card>
                            <CardHeader>
                                <CardTitle>Initial Setup</CardTitle>
                                <CardDescription>Install dependencies and configure the environment</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <p>1. Clone and set up the repository:</p>
                                    <Alert>
                                        <Terminal className="h-4 w-4" />
                                        <AlertDescription className="font-mono">
                                            git clone https://github.com/CubeStar1/Cloud-Services-API-Security.git<br />
                                            cd Cloud-Services-API-Security<br />
                                            pip install -r requirements.txt
                                        </AlertDescription>
                                    </Alert>
                                </div>
                                <div className="space-y-2">
                                    <p>2. Install AnyProxy globally:</p>
                                    <Alert>
                                        <Terminal className="h-4 w-4" />
                                        <AlertDescription className="font-mono">
                                            npm install -g anyproxy
                                        </AlertDescription>
                                    </Alert>
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Step 1: Data Collection</CardTitle>
                                <CardDescription>Two approaches for gathering cloud service traffic</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <p>Option A: Using Automated Agent</p>
                                    <Alert>
                                        <Terminal className="h-4 w-4" />
                                        <AlertDescription className="font-mono">
                                            cd data-collection/agent<br />
                                            npm install<br />
                                            cp .env.example .env<br />
                                            npm run build && npm start
                                        </AlertDescription>
                                    </Alert>
                                </div>
                                <div className="space-y-2">
                                    <p>Option B: Manual Capture</p>
                                    <Alert>
                                        <Terminal className="h-4 w-4" />
                                        <AlertDescription className="font-mono">
                                            cd data-collection/manual<br />
                                            anyproxy --port 8001 --rule general-json-key.js
                                        </AlertDescription>
                                    </Alert>
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Step 2: Initial Labeling</CardTitle>
                                <CardDescription>Generate training data using GPT-4/Gemini</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <p>Set up and run the labeling script:</p>
                                    <Alert>
                                        <Terminal className="h-4 w-4" />
                                        <AlertDescription className="font-mono">
                                            cd labelling<br />
                                            cp .env.example .env<br />
                                            # Add API keys to .env<br />
                                            python labelling.py
                                        </AlertDescription>
                                    </Alert>
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Step 3: Zero-Shot Learning</CardTitle>
                                <CardDescription>Run DeBERTa and CodeBERT models</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <p>Run DeBERTa inference:</p>
                                    <Alert>
                                        <Terminal className="h-4 w-4" />
                                        <AlertDescription className="font-mono">
                                            cd zsl/deberta<br />
                                            python inference.py
                                        </AlertDescription>
                                    </Alert>
                                </div>
                                <div className="space-y-2">
                                    <p>Run CodeBERT training and inference:</p>
                                    <Alert>
                                        <Terminal className="h-4 w-4" />
                                        <AlertDescription className="font-mono">
                                            cd zsl/codebert<br />
                                            python train.py<br />
                                            python inference.py
                                        </AlertDescription>
                                    </Alert>
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Step 4: Random Forest Training</CardTitle>
                                <CardDescription>Train the final classifier</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <Alert>
                                        <Terminal className="h-4 w-4" />
                                        <AlertDescription className="font-mono">
                                            cd rfc<br />
                                            python train.py
                                        </AlertDescription>
                                    </Alert>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>
            </Tabs>
        </div>
    )
}
