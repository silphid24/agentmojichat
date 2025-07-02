'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiChatBubbleLeftRight, HiXMark, HiMinus, HiPaperAirplane } from 'react-icons/hi2';
import { MojiCharacter } from '@/components/common';

// 메시지 타입을 정의합니다.
interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
}

const Chatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [showWelcomeMessage, setShowWelcomeMessage] = useState(true);
  
  // 채팅 관련 상태
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const chatContainerRef = useRef<HTMLDivElement>(null);

  // 메시지가 추가될 때마다 맨 아래로 스크롤합니다.
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);
  
  // 웰컴 메시지 자동 숨김 처리
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowWelcomeMessage(false);
    }, 3000); // 3초 후에 사라짐

    return () => clearTimeout(timer);
  }, []);

  // 초기 웰컴 메시지 설정
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([
        { id: 'welcome-1', text: '안녕하세요! SMHACCP 도우미 모지입니다.', sender: 'bot' },
        { id: 'welcome-2', text: '회사에 대해 궁금한 점을 무엇이든 물어보세요.', sender: 'bot' },
      ]);
    }
  }, [isOpen, messages.length]);


  const toggleChat = () => {
    setIsOpen(!isOpen);
    setShowWelcomeMessage(false); // 챗봇 열 때 웰컴 메시지 숨김
  };
  const closeChat = () => setIsOpen(false);
  const minimizeChat = () => setIsMinimized(true);
  const maximizeChat = () => setIsMinimized(false);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      text: inputValue,
      sender: 'user',
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: inputValue }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      
      const botMessage: Message = {
        id: `bot-${Date.now()}`,
        text: data.reply || '죄송합니다, 답변을 생성하는 데 문제가 발생했습니다.',
        sender: 'bot',
      };
      setMessages((prev) => [...prev, botMessage]);

    } catch (error) {
      console.error('Error fetching chatbot response:', error);
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        text: '오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
        sender: 'bot',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleQuickReply = (text: string) => {
    setInputValue(text);
    // form의 submit을 프로그래매틱하게 트리거
    const form = document.getElementById('chat-form') as HTMLFormElement;
    if (form) {
        // FormEvent를 직접 만들기가 번거로우므로, 가짜 이벤트를 만들거나,
        // handleSubmit 로직을 별도 함수로 분리하는게 더 좋습니다.
        // 여기서는 간단하게 inputValue를 채우고 유저가 전송버튼을 누르도록 유도합니다.
    }
  };

  return (
    <div className="fixed bottom-6 right-6 z-50 flex items-end">
      {/* Welcome Message */}
      <AnimatePresence>
        {!isOpen && showWelcomeMessage && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.3 }}
            className="mr-4 mb-2 p-3 bg-white rounded-lg shadow-md border"
          >
            <p className="text-sm text-gray-700 font-medium">무엇을 도와드릴까요?</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 30, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 30, scale: 0.9 }}
            transition={{ duration: 0.4, type: 'spring', stiffness: 260, damping: 20 }}
            className="absolute bg-white/95 backdrop-blur-xl rounded-3xl shadow-2xl overflow-hidden border border-gray-200/30"
            style={{
              bottom: '80px',
              right: '0px',
              width: '420px',
              height: isMinimized ? '72px' : '600px',
              maxHeight: 'calc(100vh - 120px)',
            }}
          >
            {/* Chat Header */}
            <div className="bg-white/80 backdrop-blur-sm px-6 py-4 flex justify-between items-center border-b border-gray-200/60">
               <div className="flex items-center space-x-3">
                <div className="relative">
                  <MojiCharacter size={36} />
                  <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-green-400 rounded-full border-2 border-white"></div>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 text-base">SMHACCP 모지</h3>
                  <p className="text-sm text-green-600 font-medium">● 온라인</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <button onClick={isMinimized ? maximizeChat : minimizeChat} className="w-7 h-7 rounded-full bg-yellow-400 hover:bg-yellow-500 transition-all flex items-center justify-center">
                  <HiMinus className="w-3.5 h-3.5 text-yellow-900" />
                </button>
                <button onClick={closeChat} className="w-7 h-7 rounded-full bg-red-400 hover:bg-red-500 transition-all flex items-center justify-center">
                  <HiXMark className="w-3.5 h-3.5 text-red-900" />
                </button>
              </div>
            </div>

            {/* Chat Content */}
            {!isMinimized && (
              <div className="h-[calc(100%-72px)] flex flex-col bg-gray-50/50">
                {/* Messages Area */}
                <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-6 space-y-4">
                  {messages.map((msg) => (
                    <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[80%] rounded-2xl px-4 py-2.5 ${
                          msg.sender === 'user'
                            ? 'bg-blue-500 text-white rounded-br-lg'
                            : 'bg-white text-gray-800 rounded-bl-lg border border-gray-200/80'
                        }`}
                      >
                        <p className="text-sm leading-relaxed">{msg.text}</p>
                      </div>
                    </div>
                  ))}
                  {isLoading && (
                    <div className="flex justify-start">
                        <div className="bg-white text-gray-800 rounded-2xl rounded-bl-lg border border-gray-200/80 px-4 py-2.5">
                            <div className="flex items-center space-x-1.5">
                                <span className="w-2 h-2 bg-gray-300 rounded-full animate-pulse-fast"></span>
                                <span className="w-2 h-2 bg-gray-300 rounded-full animate-pulse-medium"></span>
                                <span className="w-2 h-2 bg-gray-300 rounded-full animate-pulse-slow"></span>
                            </div>
                        </div>
                    </div>
                  )}
                </div>

                {/* Input Area */}
                <div className="p-4 bg-white/70 backdrop-blur-sm border-t border-gray-200/60">
                  {/* Quick Replies */}
                   <div className="flex flex-wrap gap-2 mb-3">
                      {['회사 소개', '복지제도', '채용정보'].map(item => (
                         <button 
                            key={item}
                            onClick={() => handleQuickReply(item)}
                            className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-medium rounded-full transition-colors duration-200"
                         >
                           {item}
                         </button>
                      ))}
                    </div>
                  <form id="chat-form" onSubmit={handleSendMessage} className="relative">
                    <input
                      type="text"
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      placeholder="어떤 도움이 필요하신가요?"
                      className="w-full pl-4 pr-12 py-3 bg-white border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-400 transition-shadow"
                      disabled={isLoading}
                    />
                    <button
                      type="submit"
                      className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 bg-blue-500 rounded-full text-white flex items-center justify-center hover:bg-blue-600 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                      disabled={isLoading || !inputValue.trim()}
                    >
                      <HiPaperAirplane className="w-5 h-5 transform -rotate-45 -translate-x-px" />
                    </button>
                  </form>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Chat Button */}
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={toggleChat}
        className="w-16 h-16 bg-blue-500 text-white rounded-full shadow-lg flex items-center justify-center relative overflow-hidden"
        style={{ zIndex: 1 }} // 말풍선 위에 버튼이 오도록 설정
      >
        <AnimatePresence initial={false}>
          <motion.div
            key={isOpen ? 'close' : 'open'}
            initial={{ rotate: -45, opacity: 0, scale: 0.5 }}
            animate={{ rotate: 0, opacity: 1, scale: 1 }}
            exit={{ rotate: 45, opacity: 0, scale: 0.5 }}
            transition={{ duration: 0.3 }}
            className="absolute flex items-center justify-center"
          >
            {isOpen ? <HiXMark size={28} /> : <MojiCharacter size={40} />}
          </motion.div>
        </AnimatePresence>
      </motion.button>
    </div>
  );
};

export default Chatbot;
