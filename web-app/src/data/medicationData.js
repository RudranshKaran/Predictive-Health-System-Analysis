export const medicationData = {
    "Clarithromycin": {
        description: "An antibiotic used to treat various bacterial infections, such as skin infections, respiratory infections, and stomach ulcers caused by Helicobacter pylori.",
        interactions: ["Atorvastatin", "Simvastatin", "Warfarin"],
        interactionDetails: {
            "Atorvastatin": "Increases risk of myopathy and rhabdomyolysis.",
            "Simvastatin": "Significantly increases risk of muscle damage.",
            "Warfarin": "Increases risk of bleeding."
        }
    },
    "Amlodipine": {
        description: "A calcium channel blocker used to treat high blood pressure (hypertension) and chest pain (angina). It works by relaxing blood vessels so blood can flow more easily.",
        interactions: ["Simvastatin", "Cyclosporine"],
        interactionDetails: {
            "Simvastatin": "Increased risk of myopathy. Dose of simvastatin should not exceed 20 mg daily.",
            "Cyclosporine": "May increase levels of amlodipine, requiring dose monitoring."
        }
    },
    "Simvastatin": {
        description: "A statin medication used to lower cholesterol and triglycerides in the blood. It helps prevent heart attacks and strokes.",
        interactions: ["Amlodipine", "Clarithromycin", "Grapefruit juice"],
        interactionDetails: {
            "Amlodipine": "Increased risk of myopathy. Dose of simvastatin should not exceed 20 mg daily.",
            "Clarithromycin": "Significantly increases risk of muscle damage. Concomitant use should be avoided.",
            "Grapefruit juice": "Increases levels of simvastatin, increasing risk of side effects."
        }
    },
    "Atorvastatin": {
        description: "A statin medication used to lower cholesterol and triglycerides in the blood. It is also used to lower the risk of stroke, heart attack, or other heart complications.",
        interactions: ["Clarithromycin", "Cyclosporine"],
         interactionDetails: {
            "Clarithromycin": "Increases risk of myopathy and rhabdomyolysis. Consider alternative antibiotic.",
            "Cyclosporine": "Increases levels of atorvastatin, requiring dose adjustment."
        }
    }
};
