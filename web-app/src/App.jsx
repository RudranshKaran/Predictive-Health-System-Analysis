import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import './App.css'

import ScribeIcon from './components/ScribeIcon';
import PatientSummaryIcon from './components/PatientSummaryIcon';
import TrendsIcon from './components/TrendsIcon';
import MenuIcon from './components/MenuIcon';
import ChevronLeftIcon from './components/ChevronLeftIcon';
import ClipboardUserIcon from './components/ClipboardUserIcon';
import SearchIcon from './components/SearchIcon';
import BellIcon from './components/BellIcon';
import MicIcon from './components/MicIcon';
import SparklesIcon from './components/SparklesIcon';
import ConditionTag from './components/ConditionTag';
import InfoCard from './components/InfoCard';
import Modal from './components/Modal';
import SearchableDropdown from './components/SearchableDropdown';
import PatientHistoryColumn from './components/PatientHistoryColumn';
import ActiveMedications from './components/ActiveMedications';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const DrugInteractionDetail = ({ interaction }) => (
    <div>
        <h3 className="text-xl font-bold text-red-700 mb-2">Drug Interaction Alert</h3>
        <p className="font-semibold">{interaction.drugA} + {interaction.drugB} ({interaction.risk})</p>
        <p className="text-gray-600 mt-2">{interaction.note}</p>
    </div>
);


// --- MOCK DATA (ENHANCED) ---
const mockPatients = [
  {
    id: 1, name: 'Priya Sharma', age: 34, gender: 'F', bloodGroup: 'O+', abhaId: '12-3656-7890-1234',
    status: 'Ongoing', 
    aiSummary: "Patient with a history of chronic hypertension and Type 2 Diabetes. Recent visits for viral fever. High risk for cardiovascular events; monitor BP closely. Penicillin allergy noted.",
    preExistingConditions: ['Hypertension', 'Type 2 Diabetes'],
    allergies: ['Penicillin'],
    currentMedications: ['Clarithromycin', 'Amlodipine', 'Simvastatin'],
    recurringIllnesses: ['Migraine', 'Allergic Rhinitis'],
    drugInteractions: [{ drugA: 'Amlodipine', drugB: 'Simvastatin', risk: 'Moderate', note: 'Increased risk of myopathy.' }],
    vitalTrends: { bp: [{date: 'Mar 25', value: '140/90'}, {date: 'Jun 25', value: '135/85'}, {date: 'Sep 25', value: '130/80'}] },
    visitHistory: [
      { date: '14 Sep 2025', diagnosis: 'Viral Fever, Suspected Dengue', notes: 'Patient presents with high fever, headache, and body aches. Prescribed Paracetamol.', clinic: { name: 'Apollo Clinic, Koramangala', location: { lat: 12.9352, lon: 77.6245 } } },
      { date: '02 Jun 2025', diagnosis: 'Annual Check-up', notes: 'Routine check-up. All vitals normal. Blood work ordered.', clinic: { name: 'Manipal Hospital, Old Airport Road', location: { lat: 12.9602, lon: 77.6482 } } },
      { date: '15 Mar 2025', diagnosis: 'Allergic Reaction', notes: 'Mild skin rash due to an unknown allergen. Prescribed antihistamines.', clinic: { name: 'Fortis Hospital, Bannerghatta Road', location: { lat: 12.8762, lon: 77.5954 } } },
    ]
  },
  {
    id: 2, name: 'Anand Kumar', age: 58, gender: 'M', bloodGroup: 'A+', abhaId: '23-4567-8901-2345',
    status: 'Incoming',
    aiSummary: "Middle-aged male with controlled hypertension. Recent check-up for diabetes showed slightly elevated fasting blood sugar. Low risk profile, but dietary discipline is key.",
    preExistingConditions: ['Hypertension'],
    allergies: ['None Reported'],
    currentMedications: [],
    recurringIllnesses: [],
    drugInteractions: [],
    vitalTrends: { bp: [{date: 'Jul 25', value: '132/82'}, {date: 'Aug 25', value: '130/84'}, {date: 'Sep 25', value: '130/80'}] },
    visitHistory: [
      { date: '13 Sep 2025', diagnosis: 'Hypertension Follow-up', notes: 'BP is stable at 130/80 mmHg. Continue current medication.', clinic: { name: 'Sakra World Hospital, Devarabisanahalli', location: { lat: 12.9255, lon: 77.6776 } } },
      { date: '10 Aug 2025', diagnosis: 'Diabetes Check-up', notes: 'Fasting blood sugar is slightly elevated. Advised dietary changes.', clinic: { name: 'Sakra World Hospital, Devarabisanahalli', location: { lat: 12.9255, lon: 77.6776 } } },
    ]
  },
   {
    id: 3, name: 'Sunita Devi', age: 45, gender: 'F', bloodGroup: 'B-', abhaId: '34-5678-9012-3456',
    status: 'Incoming',
    aiSummary: "Patient with a known history of Asthma and dust mite allergy. Requires immediate attention for exacerbations but is otherwise stable. No other chronic conditions reported.",
    preExistingConditions: ['Asthma'],
    allergies: ['Dust Mites'],
    currentMedications: [],
    recurringIllnesses: ['Bronchitis'],
    drugInteractions: [],
    vitalTrends: {},
    visitHistory: [
      { date: '12 Sep 2025', diagnosis: 'Asthma Exacerbation', notes: 'Patient experienced shortness of breath. Administered nebulizer.', clinic: { name: 'St. John\'s Medical College Hospital, Koramangala', location: { lat: 12.9288, lon: 77.6183 } } },
    ]
  },
  {
    id: 4, name: 'Ramesh Singh', age: 62, gender: 'M', bloodGroup: 'AB+', abhaId: '45-6789-0123-4567',
    status: 'Completed',
    aiSummary: "Post-op cataract surgery patient. Follow-up complete, vision has improved significantly. No complications reported.",
    preExistingConditions: ['Cataract (Post-Op)'],
    allergies: ['None Reported'],
    currentMedications: [],
    recurringIllnesses: [],
    drugInteractions: [],
    vitalTrends: {},
    visitHistory: [
      { date: '11 Sep 2025', diagnosis: 'Post-Op Follow-up', notes: 'Wound healing well. Vision acuity is 20/20.', clinic: { name: 'Narayana Nethralaya, Rajajinagar', location: { lat: 12.9902, lon: 77.5518 } } },
    ]
  }
];

const mockRegionalData = {
    cases: [ { disease: 'Dengue', clinic: 'Apollo Clinic', location: { lat: 12.9352, lon: 77.6245 }, count: 15 }, { disease: 'Malaria', clinic: 'Sakra World Hospital', location: { lat: 12.9255, lon: 77.6776 }, count: 8 }, { disease: 'Viral Fever', clinic: "St. John's", location: { lat: 12.9288, lon: 77.6183 }, count: 25 }, { disease: 'Dengue', clinic: 'Manipal Hospital', location: { lat: 12.9602, lon: 77.6482 }, count: 12 }, ],
    trends: { 'Dengue': [3, 5, 8, 15, 27, 40], 'Malaria': [2, 3, 2, 5, 8, 10], 'Viral Fever': [10, 15, 22, 18, 25, 35], },
    demographics: { 'Dengue': { '0-18': 30, '19-40': 50, '41-60': 15, '60+': 5 }, 'Malaria': { '0-18': 40, '19-40': 45, '41-60': 10, '60+': 5 }, }
};

// --- Sub-Components (Moved outside main App component to prevent re-renders) ---
  
  const ContextualAIPrompt = ({ text }) => (
    <div className="mt-2 p-2 bg-yellow-100 border border-yellow-300 rounded-md text-xs text-yellow-800 flex items-center">
    <span className="font-bold mr-1
    ">Expert Tip:</span> {text}
    </div>
  );

const NewNoteColumn = ({ noteText, setNoteText, handleProcessNotes, isProcessing }) => {
    const [isListening, setIsListening] = useState(false);
    const [contextualPrompt, setContextualPrompt] = useState('');
    const recognitionRef = useRef(null);

    useEffect(() => {
      if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = true;
        recognitionRef.current.interimResults = true;
        recognitionRef.current.lang = 'en-IN';

        recognitionRef.current.onresult = (event) => {
        let final_transcript = '';
          for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
            final_transcript += event.results[i][0].transcript;
          }
        }
        if (final_transcript) {
            setNoteText(prev => prev + final_transcript + '. ');
          }
        };

        recognitionRef.current.onend = () => {
          setIsListening(false);
        };
      }
  }, [setNoteText]);

    useEffect(() => {
      if (noteText.toLowerCase().includes('fever')) {
        setContextualPrompt("Regional Alert: Dengue cases are high in your area. Consider checking for retro-orbital pain.");
      } else {
        setContextualPrompt('');
      }
    }, [noteText]);

    const handleListen = () => {
      if (isListening) {
        recognitionRef.current.stop();
        setIsListening(false);
      } else {
        recognitionRef.current.start();
        setIsListening(true);
      }
    };

    return (
      <div className="w-1/3 p-4 mx-4 flex flex-col">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold" style={{color: '#2C3E50'}}>Diagnosis</h2>
          <button onClick={handleListen} title="Start/Stop Dictation" className={`p-2 rounded-full ${isListening ? 'bg-red-100 animate-pulse' : 'bg-gray-100 hover:bg-gray-200'}`}>
            <MicIcon isListening={isListening} />
          </button>
        </div>
        <textarea value={noteText} onChange={(e) => setNoteText(e.target.value)} className="w-full flex-1 p-3 text-sm bg-white rounded-md border border-gray-300 focus:ring-2 focus:ring-[#26A69A] mt-2" placeholder="Type or use voice-to-text..."></textarea>
        {contextualPrompt && <ContextualAIPrompt text={contextualPrompt} />}
        <button onClick={handleProcessNotes} disabled={isProcessing} style={{backgroundColor: '#FF6B6B'}} className="mt-4 w-full text-white font-semibold py-2.5 px-4 rounded-lg hover:opacity-90 disabled:bg-gray-400 flex items-center justify-center"> {isProcessing ? 'Processing...' : 'Process Notes →'} </button>
      </div>
    );
  };

const AIAssistedVerification = ({ isProcessing, aiResult, ddiAlerts }) => {
    const ConfidenceBadge = ({ confidence }) => {
        const percentage = (confidence * 100).toFixed(0);
        return (
            <span className="text-sm font-semibold text-green-700 bg-green-100 px-2 py-1 rounded-full">
                {percentage}%
            </span>
        );
    };

    return (
        <div className="w-1/3 bg-gray-50 p-4 rounded-lg shadow-inner overflow-y-auto">
            <h2 className="text-xl font-bold mb-4" style={{ color: '#2C3E50' }}>AI-Assisted Verification</h2>
            
            <div className="h-full space-y-4">
                {isProcessing && (
                    <div className="flex items-center justify-center h-full">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5" style={{ color: '#26A69A' }} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <p className="text-gray-600">Analyzing notes...</p>
                    </div>
                )}
                {!isProcessing && !aiResult && (
                    <div className="flex items-center justify-center h-full text-center">
                        <p className="text-gray-500">AI fields will appear here.<br />Click 'Process Notes' to begin.</p>
                    </div>
                )}
                {aiResult && (
                    <div className="flex flex-col justify-between h-full">
                        <div className="space-y-5">
                            {aiResult.missingInfo?.length > 0 && (
                                <div className="p-3 bg-orange-100 border-l-4 border-orange-400 text-orange-800">
                                    <h4 className="font-bold text-sm">Missing Information</h4>
                                    <p className="text-sm mt-1">{aiResult.missingInfo[0]}</p>
                                </div>
                            )}

                            {ddiAlerts.length > 0 && (
                               <div className="p-3 bg-red-100 border-l-4 border-red-400 text-red-800">
                                   <h4 className="font-bold text-sm">Drug-Drug Interaction Alert!</h4>
                                   {ddiAlerts.map((alert, i) => (
                                       <div key={i} className="mt-2 text-sm">
                                           <p className="font-semibold">{alert.drugA} + {alert.drugB} ({alert.severity})</p>
                                           <p>{alert.description}</p>
                                       </div>
                                   ))}
                               </div>
                            )}

                            {aiResult.symptoms?.length > 0 && (
                                <div>
                                    <h3 className="text-md font-semibold text-gray-700 mb-2">Symptoms</h3>
                                    <div className="space-y-2">
                                        {aiResult.symptoms.map((s, i) => (
                                            <div key={i} className="flex justify-between items-center bg-white p-3 rounded-lg shadow-sm text-gray-800">
                                                <p>{s.value}</p>
                                                <ConfidenceBadge confidence={s.confidence} />
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
            
                            <div>
                               <h3 className="text-md font-semibold text-gray-700 mb-2">Suggested Diagnosis</h3>
                               {aiResult.diagnosis?.length > 0 ? (
                                   <div className="space-y-2">
                                       {aiResult.diagnosis.map((d, i) => (
                                            <div key={i} className={`flex justify-between items-center bg-white p-3 rounded-lg shadow-sm border-l-4 ${d.suggestion ? 'border-blue-400' : 'border-gray-300'}`}>
                                                <div>
                                                    <p className="font-semibold text-gray-800">{d.description}</p>
                                                    {d.code && d.code !== 'N/A' && <p className="text-sm text-gray-500">({d.code})</p>}
                                                </div>
                                                {d.suggestion ? (
                                                    <span className="text-sm font-semibold text-blue-700 bg-blue-100 px-2 py-1 rounded-full">Suggested</span>
                                                ) : (
                                                    <span className="text-sm font-semibold text-gray-600 bg-gray-200 px-2 py-1 rounded-full">Consider</span>
                                                )}
                                            </div>
                                       ))}
                                   </div>
                               ) : (
                                    <div className="bg-white p-3 rounded-lg shadow-sm text-gray-500 text-sm">No diagnosis suggested based on notes.</div>
                               )}
                           </div>
            
                            <div>
                                <h3 className="text-md font-semibold text-gray-700 mb-2">Medication Suggested</h3>
                                {aiResult.medications?.length > 0 ? (
                                    <div className="space-y-2">
                                       {aiResult.medications.map((m, i) => (
                                            <div key={i} className="flex justify-between items-center bg-white p-3 rounded-lg shadow-sm">
                                               <p className="text-gray-800">{m.name} {m.dosage} - {m.frequency}</p>
                                               <ConfidenceBadge confidence={m.confidence} />
                                            </div>
                                       ))}
                                   </div>
                                ) : (
                                    <div className="bg-white p-3 rounded-lg shadow-sm text-gray-500 text-sm">None Suggested</div>
                                )}
                            </div>
                        </div>
                        <div className="pt-4 flex space-x-3 mt-4">
                            <button className="w-full bg-gray-200 text-gray-800 font-semibold py-2 px-4 rounded-lg hover:bg-gray-300">Edit</button>
                            <button className="w-full bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-blue-700">Approve & Save</button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

const ScribeAndVerifyView = ({ selectedPatient, patients, handlePatientSelect, setModalContent, noteText, setNoteText, handleProcessNotes, isProcessing, aiResult, ddiAlerts }) => ( 
    <div className="flex flex-1 p-6 bg-white"> 
        <PatientHistoryColumn 
            selectedPatient={selectedPatient} 
            patients={patients} 
            handlePatientSelect={handlePatientSelect} 
            setModalContent={setModalContent}
        /> 
        <NewNoteColumn 
            noteText={noteText} 
            setNoteText={setNoteText} 
            handleProcessNotes={handleProcessNotes} 
            isProcessing={isProcessing} 
        /> 
        <AIAssistedVerification 
            isProcessing={isProcessing} 
            aiResult={aiResult} 
            ddiAlerts={ddiAlerts} 
        /> 
    </div> 
);

const PatientQueueWidget = ({ isOpen, setOpen, activeTab, setActiveTab, patients, handlePatientSelect, selectedPatientId }) => (
     <div className="fixed bottom-4 right-4 z-40"> 
        <div className={`bg-white rounded-lg shadow-2xl border border-gray-200 transition-all duration-300 ease-in-out ${isOpen ? 'w-80 h-96' : 'w-48 h-12'}`}> 
            <button onClick={() => setOpen(!isOpen)} style={{backgroundColor: '#26A69A'}} className="w-full flex items-center justify-between px-4 py-3 text-white rounded-t-lg"> 
                <h3 className="font-semibold text-sm flex items-center"><ClipboardUserIcon className="h-5 w-5 mr-2" /> Patient Queue</h3> 
                <span>{isOpen ? '▼' : '▲'}</span> 
            </button> 
            {isOpen && ( 
                <div className="flex flex-col h-[calc(24rem-3rem)]"> 
                    <div className="flex border-b border-gray-200"> 
                        {['Incoming', 'Ongoing', 'Completed'].map(tab => ( 
                            <button key={tab} onClick={() => setActiveTab(tab)} className={`flex-1 py-2 text-sm font-medium ${activeTab === tab ? 'border-b-2 text-[#26A69A] border-[#26A69A]' : 'text-gray-500'}`}> {tab} </button> 
                        ))} 
                    </div> 
                    <div className="overflow-y-auto p-2"> 
                        {patients.filter(p => p.status === activeTab).map(p => ( 
                            <button key={p.id} onClick={() => handlePatientSelect(p.id)} className={`w-full text-left p-2 my-1 rounded-md ${selectedPatientId === p.id ? 'bg-teal-50' : 'hover:bg-gray-100'}`}> 
                                <p className={`font-semibold text-sm ${selectedPatientId === p.id ? 'text-teal-700' : 'text-gray-800'}`}>{p.name}</p> 
                                <p className="text-xs text-gray-500">ABHA: ...{p.abhaId.slice(-4)}</p> 
                            </button> 
                        ))} 
                    </div> 
                </div> 
            )} 
        </div> 
    </div>
);

const SimpleLineChart = ({ data, title, color = '#000000' }) => { 
    const points = useMemo(() => { 
        if (!data || data.length < 2) return ""; 
        const values = data.map(d => parseInt(String(d.value).split('/')[0])); 
        const minVal = Math.min(...values) - 5; 
        const maxVal = Math.max(...values) + 5; 
        const width = 300; 
        const height = 100; 
        return data.map((d, i) => { 
            const y = height - ((parseInt(String(d.value).split('/')[0]) - minVal) / (maxVal - minVal)) * height; 
            const x = (i / (data.length - 1)) * width; 
            return `${x},${y}`; 
        }).join(' '); 
    }, [data]); 

    if (!data || data.length === 0) return <div className="text-center text-gray-500 text-sm py-10">No trend data available</div>; 

    return ( 
        <div> 
            <h4 className="text-sm font-semibold text-gray-600">{title}</h4> 
            <svg viewBox="0 0 300 100" className="w-full h-auto"> 
                {points && <polyline fill="none" stroke={color} strokeWidth="2" points={points} />} 
                {points && data.map((d, i) => { 
                    const point = points.split(' ')[i]; if (!point) return null; 
                    const y = point.split(',')[1]; 
                    const x = (i / (data.length - 1)) * 300; 
                    return <circle key={i} cx={x} cy={y} r="3" fill={color} />; 
                })} 
            </svg> 
            <div className="flex justify-between text-xs text-gray-500 mt-1"> 
                {data.map(d => <span key={d.date}>{d.date}</span>)} 
            </div> 
        </div> 
    ); 
};

const SimpleBarChart = ({ data, color = '#000000' }) => { 
      const entries = useMemo(() => data ? Object.entries(data) : [], [data]); 
      const maxValue = useMemo(() => data && Object.values(data).length > 0 ? Math.max(...Object.values(data)) : 0, [data]); 

      if (!data || Object.keys(data).length === 0) return <div className="text-center text-gray-500 text-sm py-10">No data available</div>; 
      
      return ( 
          <div className="w-full h-48 p-4 border border-gray-200 rounded-lg bg-gray-50"> 
              <div className="flex items-end justify-around h-full"> 
                  {entries.map(([label, value]) => ( 
                      <div key={label} className="flex flex-col items-center w-1/5 h-full pt-2"> 
                          <div className="text-xs font-bold text-gray-700">{value}%</div> 
                          <div className="flex-grow w-full flex items-end"> 
                              <div className="w-3/4 mx-auto rounded-t" style={{ height: `${maxValue > 0 ? (value / maxValue) * 100 : 0}%`, backgroundColor: color }}></div> 
                          </div> 
                          <div className="mt-2 text-xs text-gray-500">{label}</div> 
                      </div> 
                  ))} 
              </div> 
          </div> 
      ); 
};

const MapComponent = ({ heatMapData }) => { const mapBounds = { minLat: 12.85, maxLat: 13.05, minLon: 77.55, maxLon: 77.75 }; const convertLocationToPercent = ({ lat, lon }) => { const top = 100 - ((lat - mapBounds.minLat) / (mapBounds.maxLat - mapBounds.minLat)) * 100; const left = ((lon - mapBounds.minLon) / (mapBounds.maxLon - mapBounds.minLon)) * 100; return { top: `${top}%`, left: `${left}%` }; }; const maxCount = Math.max(...heatMapData.map(d => d.count)); const heatMapCenter = heatMapData.find(d => d.count === maxCount); return ( <div className="relative w-full h-80 bg-teal-50 rounded-lg overflow-hidden border-2 border-teal-100"> <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(0,0,0,0.03)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,0,0,0.03)_1px,transparent_1px)] bg-[size:2rem_2rem]"></div> {heatMapCenter && ( <div className="absolute rounded-full" style={{ ...convertLocationToPercent(heatMapCenter.location), width: '200px', height: '200px', transform: 'translate(-50%, -50%)', background: 'radial-gradient(circle, rgba(255, 107, 107, 0.6) 0%, rgba(255, 107, 107, 0) 70%)', pointerEvents: 'none' }} ></div> )} {heatMapData.map((data, index) => { const { top, left } = convertLocationToPercent(data.location); return ( <div key={index} className="absolute group" style={{ top, left, transform: 'translate(-50%, -50%)' }}> <div className="w-4 h-4 rounded-full border-2 border-white cursor-pointer animate-pulse" style={{backgroundColor: '#FF6B6B'}}></div> <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1-2 w-max bg-gray-800 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap"> <p className="font-bold">{data.clinic}</p> <p>{data.disease}: {data.count} cases</p> </div> </div> ) })} </div> ); };
  
  const PatientHistoryTimeline = ({ history }) => {
    if (!history || history.length === 0) {
      return (
        <div className="bg-white p-6 rounded-lg shadow-md h-full">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Patient History Timeline</h3>
            <div className="flex items-center justify-center h-full text-gray-500">
                <p>No visit history recorded.</p>
            </div>
        </div>
      );
    }
    return (
      <div className="bg-white p-6 rounded-lg shadow-md h-full">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Patient History Timeline</h3>
        <div className="relative border-l-2 border-teal-200 ml-3 h-[calc(100%-2rem)] overflow-y-auto pr-2">
          {history.map((visit, index) => (
            <div key={index} className="mb-8 ml-8">
              <span className="absolute -left-4 flex items-center justify-center w-8 h-8 bg-teal-100 rounded-full ring-4 ring-white">
                <svg className="w-4 h-4 text-teal-600" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clipRule="evenodd"></path></svg>
              </span>
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 hover:shadow-md transition-shadow">
                <time className="text-sm font-semibold text-gray-500">{visit.date}</time>
                <h4 className="text-md font-bold text-teal-800 mt-1">{visit.diagnosis}</h4>
                <p className="text-sm text-gray-600 mt-2">{visit.notes}</p>
                <p className="text-xs text-gray-400 mt-2">at {visit.clinic.name}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

const PatientSummaryView = ({ selectedPatient, patients, handlePatientSelect, handleRegenerateSummary, isProcessing, aiResult, setModalContent }) => { 
      const [isDropdownOpen, setDropdownOpen] = useState(false);
      if(!selectedPatient) return <div className="p-8">Please select a patient.</div>; 
      
      return ( 
          <div className="p-8 grid grid-cols-1 lg:grid-cols-3 gap-6" style={{backgroundColor: '#F8F8F8'}}> 
              <div className="lg:col-span-3 bg-white p-6 rounded-lg shadow-md flex items-center relative"> 
                   <div style={{backgroundColor: '#26A69A'}} className="w-16 h-16 text-white rounded-full flex items-center justify-center text-2xl font-bold mr-4 flex-shrink-0"> {selectedPatient.name.charAt(0)} </div> 
                   <div className="flex-grow">
                        <button onClick={() => setDropdownOpen(!isDropdownOpen)} className="w-full flex justify-between items-center hover:bg-gray-100 p-2 rounded-md">
                           <div>
                                <h2 className="text-2xl font-bold text-left" style={{color: '#2C3E50'}}>{selectedPatient.name}</h2> 
                                <p className="text-sm text-gray-600 text-left">{selectedPatient.age}Y {selectedPatient.gender} • {selectedPatient.bloodGroup} • ABHA: ...{selectedPatient.abhaId.slice(-4)}</p> 
                           </div>
                           <span className={`transform transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`}>▼</span>
                        </button>
                        {isDropdownOpen && ( <SearchableDropdown options={patients} onSelect={handlePatientSelect} onClose={() => setDropdownOpen(false)} /> )}
                   </div>
              </div> 
              <div className="lg:col-span-3 bg-teal-50 border border-teal-200 text-teal-900 text-sm p-4 rounded-lg shadow-sm">
                  <div className="flex justify-between items-start">
                      <div>
                          <h3 className="font-semibold mb-1">AI Generated Summary</h3>
                          <p className={isProcessing && !aiResult ? 'text-gray-400 italic' : ''}>{isProcessing && !aiResult ? 'Generating new summary...' : selectedPatient.aiSummary}</p>
                      </div>
                      <button onClick={handleRegenerateSummary} disabled={isProcessing && !aiResult} className="flex items-center text-xs bg-white text-teal-700 font-semibold py-1 px-2 rounded-md border border-teal-200 hover:bg-teal-100 disabled:opacity-50">
                          <SparklesIcon/> Regenerate
                      </button>
                  </div>
              </div> 

              <div className="lg:col-span-3 flex overflow-x-auto space-x-4 py-4">
                  <InfoCard title="Red Flag Alerts" className="flex-shrink-0 w-64"><ConditionTag text="Penicillin Allergy" color="red" /></InfoCard> 
                  <InfoCard title="Pre-existing Conditions" className="flex-shrink-0 w-64">{selectedPatient.preExistingConditions.map(c => <ConditionTag key={c} text={c} color="orange"/>)}</InfoCard> 
                  <InfoCard title="Known Allergies" className="flex-shrink-0 w-64">{selectedPatient.allergies.map(a => <ConditionTag key={a} text={a} color="orange"/>)}</InfoCard>
              </div> 

              <div className="lg:col-span-1 h-[500px]">
                 <PatientHistoryTimeline history={selectedPatient.visitHistory} />
              </div> 

              <div className="lg:col-span-2 h-[500px]">
                 <ActiveMedications 
                    medications={selectedPatient.currentMedications} 
                    setModalContent={setModalContent}
                 />
              </div>
          </div> 
      ); 
  };
  
  const RegionalTrendsView = () => {
    const [dateRange, setDateRange] = useState('Last 4 Weeks');
    const [compareDisease, setCompareDisease] = useState('Malaria');

    return (
        <div className="p-8 grid grid-cols-1 lg:grid-cols-5 gap-6" style={{backgroundColor: '#F8F8F8'}}>
            <div className="lg:col-span-5 bg-white p-4 rounded-lg shadow-md flex items-center justify-end space-x-4">
                <select value={dateRange} onChange={(e) => setDateRange(e.target.value)} className="text-sm border-gray-300 rounded-md focus:ring-teal-500 focus:border-teal-500">
                    <option>Last 4 Weeks</option>
                    <option>Last 3 Months</option>
                    <option>Last 6 Months</option>
                </select>
                <select value={compareDisease} onChange={(e) => setCompareDisease(e.target.value)} className="text-sm border-gray-300 rounded-md focus:ring-teal-500 focus:border-teal-500">
                    <option>Malaria</option>
                    <option>Viral Fever</option>
                </select>
            </div>
            <div className="lg:col-span-3 bg-white p-6 rounded-lg shadow-md">
                <h3 className="font-semibold mb-4 text-gray-700">Disease Hotspots - Bengaluru ({dateRange})</h3>
                <MapComponent heatMapData={mockRegionalData.cases} />
            </div>
            <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-md space-y-4">
                <h3 className="font-semibold text-gray-700">Weekly Case Trends</h3>
                <SimpleLineChart data={mockRegionalData.trends['Dengue'].map((v,i) => ({date: `W${i+1}`, value: `${v}`}))} title="Dengue" color="#FF6B6B" />
                <SimpleLineChart data={mockRegionalData.trends[compareDisease].map((v,i) => ({date: `W${i+1}`, value: `${v}`}))} title={compareDisease} color="#FF8C42" />
            </div>
            <div className="lg:col-span-5 bg-white p-6 rounded-lg shadow-md">
                <h3 className="font-semibold mb-4 text-gray-700">Dengue Cases by Age Group</h3>
                <SimpleBarChart data={mockRegionalData.demographics['Dengue']} color="#26A69A" />
            </div>
        </div>
    );
  };
  
// --- Main App Layout Components (Moved outside App to prevent re-renders) ---

const Sidebar = ({ isSidebarCollapsed, setSidebarCollapsed, activeView, setActiveView }) => {
    const newTeal = '#26A69A'; const darkTeal = '#00897B'; const lightTeal = '#4DB6AC';
    return (
        <aside style={{backgroundColor: newTeal}} className={`text-white flex flex-col fixed h-full shadow-lg transition-all duration-300 ease-in-out ${isSidebarCollapsed ? 'w-20' : 'w-64'}`}>
            <div className="flex items-center justify-between px-4 py-3 border-b" style={{borderColor: darkTeal}}>
                {!isSidebarCollapsed && <Link to="/"><h1 className="text-2xl font-bold text-white tracking-wider">Aignosis</h1></Link>}
                <button onClick={() => setSidebarCollapsed(!isSidebarCollapsed)} className="p-2 rounded-md" style={{'--hover-bg': lightTeal}} onMouseOver={e => e.currentTarget.style.backgroundColor=lightTeal} onMouseOut={e => e.currentTarget.style.backgroundColor='transparent'}> {isSidebarCollapsed ? <MenuIcon /> : <ChevronLeftIcon />} </button>
            </div>
            <nav className="flex-1 px-2 py-4 space-y-2">
                {[ {name: 'Scribe & Verify', icon: ScribeIcon}, {name: 'Patient Summary', icon: PatientSummaryIcon}, {name: 'Regional Trends', icon: TrendsIcon} ].map(item => (
                    <a href="#" key={item.name} onClick={() => setActiveView(item.name)} title={item.name} style={{'--active-bg': darkTeal, '--hover-bg': lightTeal}} className={`flex items-center p-3 text-sm font-medium rounded-md transition-colors ${isSidebarCollapsed ? 'justify-center' : ''} ${activeView === item.name ? 'text-white' : ''}`} onMouseOver={e => {if(activeView !== item.name) e.currentTarget.style.backgroundColor=lightTeal}} onMouseOut={e => {if(activeView !== item.name) e.currentTarget.style.backgroundColor='transparent'}} ref={el => { if(el) el.style.backgroundColor = activeView === item.name ? darkTeal : 'transparent'; }} >
                        <item.icon className={isSidebarCollapsed ? "h-6 w-6" : "h-5 w-5 mr-3"}/>
                        {!isSidebarCollapsed && item.name}
                    </a>
                ))}
            </nav>
            <div className="px-4 py-4 border-t" style={{borderColor: darkTeal}}>
                 <div className="flex items-center">
                    <div style={{backgroundColor: '#2C3E50'}} className="w-10 h-10 rounded-full flex items-center justify-center font-bold text-gray-200">DK</div>
                    {!isSidebarCollapsed && <div className="ml-3"><p className="text-sm font-semibold">Dr. Kapoor</p><p className="text-xs text-teal-100">Cardiologist</p></div>}
                </div>
            </div>
        </aside>
      );
};

const Header = ({ activeView }) => (
    <header className="bg-white py-2 px-6 flex items-center justify-between border-b border-gray-200">
        <div className="flex items-center">
            <h1 className="text-xl font-bold" style={{color: '#2C3E50'}}>{activeView}</h1>
        </div>
        <div className="w-full max-w-md mx-4">
            <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"> <SearchIcon /> </div>
                <input type="text" placeholder="Search Patient by Name or ABHA ID..." className="block w-full bg-gray-100 border border-gray-200 rounded-md py-2 pl-10 pr-3 text-sm placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-teal-500 focus:border-teal-500" />
            </div>
        </div>
        <div className="flex items-center space-x-4">
            <button className="relative p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full"> <BellIcon /> <span className="absolute top-0 right-0 h-3 w-3 bg-red-500 rounded-full border-2 border-white"></span> </button>
            <div className="w-px h-6 bg-gray-200"></div>
            <div className="flex items-center"> <p className="text-sm text-gray-500">{new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}</p> </div>
        </div>
    </header>
);

const MainContent = ({ activeView, ...props }) => {
    switch (activeView) {
      case 'Scribe & Verify': 
        return <ScribeAndVerifyView {...props} />;
      case 'Patient Summary': 
        return <PatientSummaryView {...props} />;
      case 'Regional Trends': 
        return <RegionalTrendsView />;
      default: 
        return <ScribeAndVerifyView {...props} />;
    }
};

// --- Main App Component ---
export default function App() {
  const [activeView, setActiveView] = useState('Scribe & Verify');
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [patients, setPatients] = useState(mockPatients);
  const [isSidebarCollapsed, setSidebarCollapsed] = useState(true); // Default to collapsed
  const [isQueueOpen, setQueueOpen] = useState(false); // Default to collapsed
  const [activeQueueTab, setActiveQueueTab] = useState('Incoming');
  const [modalContent, setModalContent] = useState(null);
  const [noteText, setNoteText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  const [ddiAlerts, setDdiAlerts] = useState([]);
  const selectedPatient = useMemo(() => patients.find(p => p.id === selectedPatientId), [selectedPatientId, patients]);

  useEffect(() => {
    if (selectedPatientId) {
        const fetchPatientData = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/patient_summary/${selectedPatientId}`);
                if (response.ok) {
                    const data = await response.json();
                    // Here you would update your state with the fetched data.
                    // For now, we'll just log it to the console.
                    console.log('Fetched patient data:', data);
                }
            } catch (error) {
                console.error("Error fetching patient data:", error);
            }
        };
        fetchPatientData();
    }
  }, [selectedPatientId]);

  const handleProcessNotes = async () => {
    if (!noteText.trim() || !selectedPatient) return;
    setIsProcessing(true);
    setAiResult(null);
    setDdiAlerts([]);

    let analysisResult = null;

    try {
        // First, try the MedGemma endpoint
        const medgemmaResponse = await fetch(`${API_BASE_URL}/medgemma_analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: noteText }),
        });

        if (medgemmaResponse.ok) {
            analysisResult = await medgemmaResponse.json();
        } else {
            // If MedGemma fails, throw an error to trigger the fallback
            throw new Error('MedGemma failed, trying fallback.');
        }
    } catch (error) {
        console.warn(error.message);
        try {
            // Fallback to Gemini endpoint
            const geminiResponse = await fetch(`${API_BASE_URL}/gemini_analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: noteText, 
                    region: 'Bengaluru', // Hardcoding region for now
                    current_medications: selectedPatient.currentMedications || []
                }),
            });

            if (geminiResponse.ok) {
                analysisResult = await geminiResponse.json();
            } else {
                const errorData = await geminiResponse.json();
                console.error("Gemini fallback also failed:", errorData.error);
                // Optionally, set an error state to show in the UI
            }
        } catch (fallbackError) {
            console.error("Error with Gemini fallback:", fallbackError);
        }
    }

    if (analysisResult) {
        setAiResult(analysisResult);

        if (analysisResult.medications && analysisResult.medications.length > 0) {
            const newMedications = analysisResult.medications.map(m => m.name);
            try {
                const ddiResponse = await fetch(`${API_BASE_URL}/ddi_check`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        current_medications: selectedPatient.currentMedications,
                        new_medications: newMedications
                    })
                });
                if (ddiResponse.ok) {
                    const ddiData = await ddiResponse.json();
                    if (ddiData.interactions) {
                        setDdiAlerts(ddiData.interactions);
                    }
                }
            } catch (error) {
                console.error("Error checking DDI:", error);
            }
        }
    }

    setIsProcessing(false);
  };
  
  const handleRegenerateSummary = async () => {
    if (!selectedPatient) return;
    setIsProcessing(true);
    let newSummary = "Failed to generate summary due to an error."; // Default error message
    try {
        const response = await fetch(`${API_BASE_URL}/generate_summary`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(selectedPatient)
        });

        if (response.ok) {
            const data = await response.json();
            newSummary = data.summary;
        } else {
            const errorData = await response.json();
            console.error("Error generating summary:", errorData.error);
        }
    } catch (error) {
        console.error("Error generating summary:", error);
    }

    setPatients(prevPatients => prevPatients.map(p => 
        p.id === selectedPatientId ? { ...p, aiSummary: newSummary } : p
    ));
    setIsProcessing(false);
  };

  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = `@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'); body { font-family: 'Inter', sans-serif; }`;
    document.head.appendChild(style);
  }, []);

  const handlePatientSelect = (id) => { setSelectedPatientId(id); setNoteText(''); setAiResult(null); };
  
  return (
    <div className="min-h-screen font-sans text-gray-700 flex" style={{fontFamily: "'Inter', sans-serif", backgroundColor: '#F8F8F8'}}>
      {modalContent && <Modal content={modalContent} onClose={() => setModalContent(null)} />}
      <Sidebar 
        isSidebarCollapsed={isSidebarCollapsed} 
        setSidebarCollapsed={setSidebarCollapsed} 
        activeView={activeView} 
        setActiveView={setActiveView} 
      />
      <main className={`flex-1 flex flex-col transition-all duration-300 ease-in-out ${isSidebarCollapsed ? 'ml-20' : 'ml-64'}`}>
        <Header activeView={activeView} />
        <div className="flex-1 overflow-y-auto relative">
          <MainContent 
            activeView={activeView}
            selectedPatient={selectedPatient} 
            patients={patients} 
            handlePatientSelect={handlePatientSelect} 
            setModalContent={setModalContent} 
            noteText={noteText} 
            setNoteText={setNoteText} 
            handleProcessNotes={handleProcessNotes} 
            isProcessing={isProcessing} 
            aiResult={aiResult} 
            ddiAlerts={ddiAlerts}
            handleRegenerateSummary={handleRegenerateSummary}
          />
          {activeView === 'Scribe & Verify' && 
            <PatientQueueWidget 
              isOpen={isQueueOpen} 
              setOpen={setQueueOpen} 
              activeTab={activeQueueTab} 
              setActiveTab={setActiveQueueTab} 
              patients={patients} 
              handlePatientSelect={handlePatientSelect} 
              selectedPatientId={selectedPatientId} 
            />
          }
        </div>
      </main>
    </div>
  );
}

