Page({
  data: {
    // 这里填入您的公网访问地址！！
    // 如果还没有部署，可以先用内网穿透地址
    webUrl: "https://your-domain.com/"
  },

  onLoad(options) {
    console.log("页面加载完成");
  },

  bindMessage(e) {
    console.log("收到 WebView 消息:", e.detail);
  }
})
