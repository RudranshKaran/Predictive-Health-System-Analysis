import React from 'react';

const InfoCard = ({ title, children }) => ( <div className="bg-white p-3 rounded-lg shadow-sm border border-gray-200"> <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">{title}</h3> <div className="flex flex-wrap gap-2">{children}</div> </div> );

export default InfoCard;
