// Mobile nav: show on scroll
(function () {
  const nav = document.querySelector(".site-nav")
  if (!nav) return
  function update() {
    nav.classList.toggle("nav--scrolled", window.scrollY > 60)
  }
  window.addEventListener("scroll", update, { passive: true })
  update()
})()

// Schedule day tab switching
document.addEventListener("DOMContentLoaded", () => {
  const tabs = document.querySelectorAll(".schedule-tab")
  const days = document.querySelectorAll(".schedule-day")

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.day
      tabs.forEach((t) => {
        t.classList.remove("active")
        t.setAttribute("aria-selected", "false")
      })
      days.forEach((d) => d.classList.remove("active"))
      tab.classList.add("active")
      tab.setAttribute("aria-selected", "true")
      document.querySelector(`.schedule-day[data-day="${target}"]`)?.classList.add("active")
    })
  })

  // Smooth scroll for nav anchor links
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
      const target = document.querySelector(link.getAttribute("href"))
      if (!target) return
      e.preventDefault()
      target.scrollIntoView({ behavior: "smooth", block: "start" })
      history.replaceState(null, "", link.getAttribute("href"))
    })
  })
})
