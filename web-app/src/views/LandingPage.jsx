import React from 'react';
import { Link } from 'react-router-dom';
import DataWeaveAnimation from '../components/DataWeaveAnimation';
import useOnScreen from '../hooks/useOnScreen';

const AnimatedSection = ({ children, className = '' }) => {
    const [ref, isVisible] = useOnScreen({ threshold: 0.1, triggerOnce: false });
    return (
        <section ref={ref} className={`h-screen snap-start flex flex-col items-center justify-center p-4 md:px-8 lg:px-16 ${className} ${isVisible ? 'is-visible' : ''}`}>
            {children}
        </section>
    );
};

const LandingPage = () => {
    return (
        <div className="text-black font-sans" style={{ fontFamily: "'Inter', sans-serif" }}>
            <DataWeaveAnimation />
            <div className="relative z-10 h-screen overflow-y-scroll snap-y snap-mandatory">
                <header className="absolute top-0 left-0 w-full p-4 flex justify-between items-center bg-transparent">
                    <div className="flex items-center">
                        <h1 className="text-2xl font-bold tracking-wider text-[#26A69A]">Aignosis</h1>
                    </div>
                    <nav className="flex items-center space-x-4">
                        <span className="text-sm text-[#26A69A]">{new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}</span>
                    </nav>
                </header>

                <section className="h-screen snap-start flex flex-col items-center justify-center text-center p-4">
                    <main className="space-y-8">
                        <h2 className="text-5xl md:text-6xl font-bold text-gray-800" style={{ textShadow: '0 1px 2px rgba(0,0,0,0.1)' }}>
                            Intelligent Diagnostics for Modern Healthcare
                        </h2>
                        <p className="text-lg md:text-xl max-w-3xl mx-auto text-gray-600">
                            Aignosis leverages cutting-edge AI to provide predictive health insights, drug interaction alerts, and regional trend analysis to support clinicians and improve patient outcomes.
                        </p>
                        <Link
                            to="/dashboard"
                            className="bg-teal-700 text-white font-semibold py-3 px-8 rounded-full text-lg hover:bg-teal-600 transition-transform transform hover:scale-105 inline-block shadow-lg"
                            style={{ backgroundColor: '#26A69A', color: '#ffffff' }}
                        >
                            Enter Dashboard
                        </Link>
                    </main>
                </section>

                <AnimatedSection>
                    <div className="max-w-6xl mx-auto text-center">
                        <h3 className="text-4xl font-extrabold bg-gradient-to-r from-teal-500 to-cyan-400 bg-clip-text text-transparent mb-4 fade-in-up">What We Do?</h3>
                        <p className="text-lg text-gray-600 mb-12 fade-in-up delay-1">
                            Aignosis is a health intelligence platform that gives doctors a clear, <span className="font-bold text-teal-600 underline transition-transform inline-block hover:scale-105">concise view of patient information</span>. By integrating <span className="font-bold text-teal-600 underline transition-transform inline-block hover:scale-105">data from ABHA records</span>, our system summarizes critical details such as medical history, chronic conditions, allergies, vaccinations, and prescriptions, helping doctors focus on what matters most in limited consultation time.
                            <br/><br/>
                            Beyond individual care, Aignosis analyzes anonymized clinic data to detect emerging health trends within regions and to <span className="font-bold text-teal-600 underline transition-transform inline-block hover:scale-105">flag potential drug–drug interactions</span> or side effects from medicines and vaccines. This dual approach not only supports faster, safer clinical decisions but also enables proactive public health interventions, transforming healthcare from reactive to preventive.
                        </p>
                    </div>
                </AnimatedSection>

                <AnimatedSection>
                    <div className="max-w-4xl mx-auto text-center">
                        <h4 className="text-3xl font-extrabold bg-gradient-to-r from-teal-500 to-cyan-400 bg-clip-text text-transparent mb-4 fade-in-up">AI-Powered Patient Summaries</h4>
                        <p className="text-gray-600 mb-8 fade-in-up delay-1">
                            Instantly generate concise, clinically relevant summaries of a patient's history. Our AI extracts key information from visit notes, lab results, and more, highlighting chronic conditions, allergies, and potential risks, allowing you to get up to speed in seconds.
                        </p>
                        <img src="/patient_summary_placeholder.svg" alt="Patient Summary UI" className="rounded-lg shadow-xl bg-white p-2 border fade-in-up delay-2 mx-auto" />
                    </div>
                </AnimatedSection>

                <AnimatedSection>
                    <div className="max-w-4xl mx-auto text-center">
                        <h4 className="text-3xl font-extrabold bg-gradient-to-r from-teal-500 to-cyan-400 bg-clip-text text-transparent mb-4 fade-in-up">Real-Time Regional Analysis</h4>
                        <p className="text-gray-600 mb-8 fade-in-up delay-1">
                            Monitor public health trends as they emerge. Our platform analyzes anonymized data to identify disease hotspots, track weekly case trends, and provide demographic insights, empowering proactive healthcare strategies.
                        </p>
                        <img src="/regional_analysis_placeholder.svg" alt="Regional Analysis UI" className="rounded-lg shadow-xl bg-white p-2 border fade-in-up delay-2 mx-auto" />
                    </div>
                </AnimatedSection>

                <AnimatedSection className="relative">
                    <div className="max-w-4xl mx-auto text-center">
                        <h3 className="text-4xl font-extrabold bg-gradient-to-r from-teal-500 to-cyan-400 bg-clip-text text-transparent mb-4 fade-in-up">Our Future Goals</h3>
                        <p className="text-lg text-gray-600 mb-4 fade-in-up delay-1">
                            We are committed to continuous innovation. Our next major milestone is the integration of <span className="font-bold text-teal-600 underline transition-transform inline-block hover:scale-105">Biomarker Analysis</span>. This will enable clinicians to track and analyze critical biomarkers over time, providing deeper insights into disease progression and treatment efficacy, paving a way for truly predictive and personalized medicine.
                        </p>
                    </div>
                    <footer className="w-full p-8 text-center absolute bottom-0 left-0">
                        <p className="text-sm text-gray-500">&copy; {new Date().getFullYear()} Aignosis. All rights reserved.</p>
                    </footer>
                </AnimatedSection>
            </div>
        </div>
    );
};

export default LandingPage;
