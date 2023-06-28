import React from 'react'

const InfoCard = ({ data }) => {

    const text_truncate = (str, length, ending) => {
        if (length == null) {
            length = 180;
        }
        if (ending == null) {
            ending = '...';
        }
        if (str.length > length) {
            return str.substring(0, length - ending.length) + ending;
        } else {
            return str;
        }
    };



    return (
        <div className="max-w-sm rounded overflow-hidden flex flex-col justify-between hover:bg-gray-100 border m-1">
            <div className="px-3 py-2">
                <div className="font-bold text-xl mb-2 truncate">{data.desc}</div>
                <p className="text-gray-700 text-base">
                    {text_truncate(data.attchmntText)}
                </p>
            </div>
            <div className="px-6 pt-4 pb-2 flex justify-between">
                <p className="inline-block bg-gray-200 rounded px-3 py-1 text-sm font-semibold text-gray-700 mr-2 mb-2">{data.an_dt}</p>

                <div className="bg-gray-200 rounded px-2 py-1 text-sm font-semibold text-gray-700 mr-2 mb-2 flex justify-between">
                    <svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 64 64" className='w-5 h-5 mx-1' viewBox="0 0 64 64" id="pdf-file"><path fill="#eff2f3" d="M58.2,3.2l0,49H5.8l0-38.4L19.6,0l35.5,0C56.8,0,58.2,1.5,58.2,3.2z"></path><path fill="#ff0000" d="M16.7,13.8l-10.8,0L19.6,0l-0.1,10.8C19.6,12.5,18.3,13.8,16.7,13.8z"></path><path fill="#ff0000" d="M37.1 28c-2.9-2.8-5-6-5-6 .8-1.2 2.7-8.1-.2-10.1-2.9-2-4.4 1.7-4.4 1.7-1.3 4.6 1.8 8.8 1.8 8.8l-3.5 7.7c-.4 0-11.5 4.3-7.7 9.6 3.9 5.3 9.3-7.5 9.3-7.5 2.1-.7 8.5-1.6 8.5-1.6 2.5 3.4 5.5 4.5 5.5 4.5 4.6 1.2 5.1-2.6 5.1-2.6C46.6 27.5 37.1 28 37.1 28zM20 37.9c-.1 0-.1-.1-.1-.1-.6-1.4 4-4.1 4-4.1S21.4 38.5 20 37.9zM30.4 13.7c1.3 1.2.2 5.3.2 5.3S29.1 14.9 30.4 13.7zM29.2 29.2l1.8-4.4 2.8 3.4L29.2 29.2zM44 32.4C44 32.4 44 32.4 44 32.4L44 32.4 44 32.4c-.8 1.3-4.1-1.5-4.4-1.8l0 0c0 0 0 0 0 0 0 0 0 0 0 0l0 0C40.1 30.6 44.4 30.9 44 32.4L44 32.4zM58.2 49l0 11.8c0 1.7-1.4 3.2-3.2 3.2L8.9 64c-1.7 0-3.2-1.4-3.2-3.2l0-11.8H58.2z"></path><path fill="#eff2f3" d="M27.9 54.2c0 .8-.3 1.4-.8 1.9-.5.4-1.3.6-2.3.6h-.8v2.9h-1.3v-7.7H25c1 0 1.7.2 2.2.6C27.7 52.9 27.9 53.4 27.9 54.2zM24.1 55.7h.7c.6 0 1.1-.1 1.4-.3.3-.2.5-.6.5-1.1 0-.4-.1-.8-.4-1-.3-.2-.7-.3-1.3-.3h-.9V55.7zM35.8 55.7c0 1.3-.4 2.3-1.1 2.9-.7.7-1.7 1-3.1 1h-2.2v-7.7h2.4c1.2 0 2.2.3 2.9 1C35.4 53.5 35.8 54.5 35.8 55.7zM34.4 55.7c0-1.9-.9-2.8-2.6-2.8h-1.1v5.6h.9C33.5 58.6 34.4 57.6 34.4 55.7zM38.7 59.6h-1.3v-7.7h4.4v1.1h-3.1v2.4h2.9v1.1h-2.9V59.6z"></path></svg>
                    <a href={data.attchmntFile} target="_blank">Find Attachment</a>
                </div>
            </div>
        </div>
    )
}

export default InfoCard
