from abc import ABC, abstractmethod
from typing_extensions import override

class BaseInstructionHandler(ABC):
    def __init__(self):
        self.ate = {}
        self.atsc = {}
        self.aspe = {}
        self.aooe = {}
        self.aope = {}
        self.aoste = {}
        self.instruct = {
                "ate": self.ate,
                "atsc": self.atsc,
                "aspe": self.aspe,
                "aooe": self.aooe,
                "aope": self.aope,
                "aoste": self.aoste
        }
        ############################## ATE ############################## 
        self.ate["definition"]= """Đầu ra sẽ là các từ ngữ khía cạnh (aspect terms) xuất hiện trực tiếp trong văn bản đầu vào với prefix [ate] ở đầu output. Nếu tìm thấy nhiều khía cạnh, hãy ngăn cách chúng bằng " ## " . Trong trường hợp không có khía cạnh nào, đầu ra là none .\n"""
        self.ate["delim_instruct"] = "Hãy hoàn thành ví dụ sau:\ninput: "
        self.ate["eos_instruct"] = "\noutput: "

        ############################## ATSC ############################## 
        self.atsc["definition"] = """Đầu ra sẽ là POS nếu tích cực, NEG nếu tiêu cực hoặc NEU nếu trung tính dựa trên cảm xúc của khía cạnh được xác định trong đầu vào kèm prefix [atsc] ở đầu output. Lưu ý rằng khía cạnh này có thể xuất hiện trực tiếp hoặc là khía cạnh ẩn (<IA>) được suy luận từ ngữ cảnh. Nếu có nhiều sentiment thì mỗi sentiment được ngăn cách bởi " ## " . Nếu không có cảm xúc nào thì trả về none.\n"""
        self.atsc["delim_instruct"] = "Hãy hoàn thành ví dụ sau:\ninput: "
        self.atsc["eos_instruct"] = "\noutput: "

        ############################## ASPE ############################## 
        self.aspe["definition"] = """Đầu ra sẽ là các cặp bao gồm khía cạnh (cả hiện hữu và ẩn) cùng với cực tính cảm xúc của chúng kèm prefix [aspe] ở đầu output. Cực tính cảm xúc là POS nếu tích cực, NEG nếu tiêu cực hoặc NEU nếu trung tính. Định dạng quy định cho mỗi cặp là: khía cạnh $ cảm xúc . Nếu có nhiều cặp, hãy ngăn cách chúng bằng " ## " . Nếu không có khía cạnh nào, đầu ra là none $ none . Nếu là khía cạnh ẩn thì aspect sẽ là <IA>.\n"""
        self.aspe["delim_instruct"] = "Hãy hoàn thành ví dụ sau:\ninput: "
        self.aspe["eos_instruct"] = "\noutput: "

        ############################## AOOE ############################## 
        self.aooe["definition"] = """Đầu ra sẽ là các từ ngữ quan điểm hoặc từ mô tả liên quan đến khía cạnh được chỉ định trong câu (khía cạnh này có thể là tường minh hoặc ẩn) kèm prefix [aooe] ở đầu output. Nếu có nhiều từ quan điểm cho cùng một khía cạnh, hãy ngăn cách chúng bằng " ## " . Nếu không tìm thấy, đầu ra là none \n"""
        self.aooe["delim_instruct"] = "Hãy hoàn thành ví dụ sau:\ninput: "
        self.aooe["eos_instruct"] = "\noutput: "

        ############################## AOPE ############################## 
        self.aope["definition"] = """Đầu ra sẽ là các cặp bao gồm khía cạnh (cả hiện hữu và ẩn) đi kèm với từ ngữ quan điểm mô tả cho nó kèm prefix [aope] ở đầu output. Định dạng quy định cho mỗi cặp là: khía cạnh $ từ quan điểm . Nếu có nhiều cặp, hãy ngăn cách chúng bằng " ## " . Khía cạnh ẩn sẽ được thể hiện là <IA> . Nếu không tìm thấy cặp nào thì trả về none $ none \n"""
        self.aope["delim_instruct"] = "Hãy hoàn thành ví dụ sau:\ninput: "
        self.aope["eos_instruct"] = "\noutput: "

        ############################## AOSTE ############################## 
        self.aoste["definition"] = """Đầu ra sẽ là các bộ ba bao gồm: thuật ngữ khía cạnh (cả hiện hữu và ẩn), từ ngữ quan điểm mô tả cho nó, và cực tính cảm xúc tương ứng (POS nếu tích cực, NEG nếu tiêu cực và NEU nếu trung tính) kèm prefix [aoste] ở đầu output. Định dạng quy định cho mỗi bộ ba là: khía cạnh $ từ quan điểm $ cảm xúc . Nếu có nhiều bộ ba, hãy ngăn cách chúng bằng " ## " \n"""
        self.aoste["delim_instruct"] = "Hãy hoàn thành ví dụ sau:\ninput: "
        self.aoste["eos_instruct"] = "\noutput: "
	
    def get_task_handler(self, task):
        task = task.lower()
        instruct = self.instruct.get(task, None)
        if not instruct:
            raise ValueError(f"Task {task} not found")
        return instruct

    def apply_instruction(self, text, task):
        task_handler = self.get_task_handler(task)
        instruction = task_handler["instruction"] + text + task_handler["eos_instruct"]
        return instruction

    def load_instruction_0(self):
        """Không có example"""
        ############################## ATE ##############################
        self.ate["instruction"] = self.ate["definition"] + self.ate["delim_instruct"]

        ############################## ATSC ##############################
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["delim_instruct"]

        ############################## ASPE ##############################
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["delim_instruct"]

        ############################## AOOE ##############################
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["delim_instruct"]

        ############################## AOPE ##############################
        self.aope["instruction"] = self.aope["definition"] + self.aope["delim_instruct"]

        ############################## AOSTE ############################## 
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["delim_instruct"]


    @abstractmethod
    def load_instruction_1(self):
        raise NotImplementedError

    @abstractmethod
    def load_instruction_2(self):
        raise NotImplementedError

    @abstractmethod
    def load_instruction_2_modified(self):
        raise NotImplementedError


class InstructionHandler(BaseInstructionHandler):

    @override
    def load_instruction_1(self):
        """ Instruction với 2 posivie example """
        
        ############################## ATE ##############################
        self.ate["example"] = """Ví dụ 1:
input: sản phẩm giao khá nhanh và đóng gói cẩn thận nội dung khá gay cấn
output: [ate] đóng gói ## giao ## nội dung
Ví dụ 2:
input: Sản phẩm đúng theo ảnh trên shop, chất lượng rất tốt
output: [ate] chất lượng ## Sản phẩm
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ############################## ATSC ##############################
        self.atsc["example"] = """Ví dụ 1:
input: Kéo béng thật, cầm chắc tay, tôi đã mua thử 1 cái, ok quá đã mua thêm 1 cái ## Aspect: Kéo
output: [atsc] POS
Ví dụ 2:
input: Đóng gói rất chỉnh chu, rất đẹp, kỹ càng, giao hàng nhanh hơn dự kiến Sản phẩm rất tốt, da dày, màu đẹp như hình, rất chất lượng ## Aspect: da
output: [atsc] POS
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]

        ############################## ASPE ##############################
        self.aspe["example"] = """Ví dụ 1:
input: hàng như quảng cáo.nhìn chung là dễ thương
output: [aspe] hàng $ POS ## <IA> $ POS
Ví dụ 2:
input: giao hang nhiet tinh ,dong goi can than ,san pham sai rat em ,ung ho sop thiem
output: [aspe] giao hang $ POS ## ,dong goi $ POS ## sai $ POS ## sop $ POS
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]

        ############################## AOOE ##############################
        self.aooe["example"] = """Ví dụ 1:
input: Cực kì hài lòng từ giao hàng đến chất lượng sản phẩm . ## Aspect: chất lượng sản phẩm
output: [aooe] Cực kì hài lòng
Ví dụ 2:
input: Quyển sách ít chữ và hình ảnh sinh động, phù hợp cho những bạn trong độ tuổi dưới 3, trông sách còn có hiệu éng đường hầm và cầu vòng khá thú vị nữa. ## Aspect: hình ảnh
output: [aooe] sinh động,
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ############################## AOPE ##############################
        self.aope["example"] = """Ví dụ 1:
input: hay vlinnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
output: [aope] <IA> $ hay vlinnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
Ví dụ 2:
input: Hài lòng với sản phẩm nhận được. Mong rằng tiki sẽ luôn có nhiều sản phẩm tốt.
output: [aope] sản phẩm $ Hài lòng ## sản phẩm $ tốt.
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]

        ############################## AOSTE ############################## 
        self.aoste["example"] = """Ví dụ 1:
input: Giao hàng nhanh, chất lượng quạt chạy ổn, phù hợp giá tiền.
output: [aoste] Giao hàng $ nhanh, $ POS ## quạt chạy $ ổn, $ POS ## giá tiền. $ phù hợp $ POS
Ví dụ 2:
input: Sản phẩm ok, giao hàng nhanh, dịch vụ cũng tốt, sẽ mua hàng nữa nếu cần
output: [aoste] Sản phẩm $ ok, $ POS ## giao hàng $ nhanh, $ POS ## dịch vụ $ cũng tốt, $ POS
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]

    @override		
    def load_instruction_2(self):
        """ Instruction với 2 positive example, 2 negative example và 2 neutral example """
        
        ############################## ATE ##############################
        self.ate["example"] = """Ví dụ 1:
input: sản phẩm giao khá nhanh và đóng gói cẩn thận nội dung khá gay cấn
output: [ate] đóng gói ## giao ## nội dung
Ví dụ 2:
input: Sản phẩm đúng theo ảnh trên shop, chất lượng rất tốt
output: [ate] chất lượng ## Sản phẩm
Ví dụ 3:
input: Ấm ko chắc chắn lắm, bị cọ sát xước sơn bên ngoài, miệng ấm hơi rộng, nếu rót nước gần hết thì nắp ấm há mồm chảy nước ra theo.
output: [ate] Ấm ## miệng ấm ## nắp ấm ## sơn
Ví dụ 4:
input: Đã nhận đc sản phẩm . Nhưng khi check thì k đc ạ ?
output: [ate] check
Ví dụ 5:
input: đang sử dụng nên chưa biết là có tốt hay không
output: [ate] sử dụng
Ví dụ 6:
input: Cho e hoi do choi xe qua tang con khong vay? va co do choi moi khong? de e mua hang
output: [ate] do choi xe qua tang ## do choi moi
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ############################## ATSC ##############################
        self.atsc["example"] = """Ví dụ 1:
input: Kéo béng thật, cầm chắc tay, tôi đã mua thử 1 cái, ok quá đã mua thêm 1 cái ## Aspect: cầm
output: [atsc] POS
Ví dụ 2:
input: Đóng gói rất chỉnh chu, rất đẹp, kỹ càng, giao hàng nhanh hơn dự kiến Sản phẩm rất tốt, da dày, màu đẹp như hình, rất chất lượng ## Aspect: Đóng gói
output: [atsc] POS ## POS
Ví dụ 3:
input: Phiền hà khách hàng khi phải đủ 25 kí tự ms gửi dc ## Aspect: gửi dc
output: [atsc] NEG
Ví dụ 4:
input: sp k mang lại hiệu quả như quảng cáo ## Aspect: sp
output: [atsc] NEG
Ví dụ 5:
input: Thật khó có thể trách tiki vì đợt này giao hàng muộn vì nhiều lý do, lý do nhà in chậm, vận chuyển, hệ thống lớn, tuy bản thân phải chờ khá sốt ruột rồi nhưng chuyện gì đã qua thì nên cho qua, chỉ tự trách mình là người nhận sau :3 ## Aspect: giao hàng
output: [atsc] NEU
Ví dụ 6:
input: Có chế độ sấy khô chén dĩa ko ạ? Nếu có dùng chức năng nào để có chế độ sấy khô vậy ạ? Vì mình thấy có bộ phận sấy trong hình hướng dẫn sử dụng mà ## Aspect: chế độ sấy khô
output: [atsc] NEU
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]

        ############################## ASPE ##############################
        self.aspe["example"] = """Ví dụ 1:
input: hàng như quảng cáo.nhìn chung là dễ thương
output: [aspe] hàng $ POS ## <IA> $ POS
Ví dụ 2:
input: giao hang nhiet tinh ,dong goi can than ,san pham sai rat em ,ung ho sop thiem
output: [aspe] giao hang $ POS ## ,dong goi $ POS ## sai $ POS ## sop $ POS
Ví dụ 3:
input: Gối ôm quá ngắn, cứng quá,.............................
output: [aspe] Gối ôm $ NEG ## <IA> $ NEG
Ví dụ 4:
input: sản phẩm rất tệ, dao kéo rất yếu, gọt trái cây không được
output: [aspe] sản phẩm $ NEG ## dao kéo $ NEG ## gọt trái cây $ NEG
Ví dụ 5:
input: uống cũng tạm dc viết cho đủ 50 từ mà có vẻ dài quá
output: [aspe] uống $ NEU
Ví dụ 6:
input: Sản phẩm hiện vẫn chưa sử dụng, tiki vui lòng liên hệ đổi lại đúng sp của Hoàng Long Store bán. Cảm ơn.
output: [aspe] liên hệ đổi lại $ NEU
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]

        ############################## AOOE ##############################
        self.aooe["example"] = """Ví dụ 1:
input: Cực kì hài lòng từ giao hàng đến chất lượng sản phẩm . ## Aspect: chất lượng sản phẩm
output: [aooe] Cực kì hài lòng
Ví dụ 2:
input: Quyển sách ít chữ và hình ảnh sinh động, phù hợp cho những bạn trong độ tuổi dưới 3, trông sách còn có hiệu éng đường hầm và cầu vòng khá thú vị nữa. ## Aspect: hình ảnh
output: [aooe] sinh động,
Ví dụ 3:
input: balo có 2 nút bấm để khoá lại một bên bị rớt ra mua về phải đi đóng lại . bán hàng ko kiểm tra sp trước khi giao cho kh ## Aspect: bán hàng
output: [aooe] ko kiểm tra sp
Ví dụ 4:
input: Sau gần 1h loay hoay cuối cùng mình cũng pair được. Sau đây xin chia sẻ các bước cho ai lần đầu tiên hoặc chưa pair được: Bước 1. bạn phải tắt cả 2 tai nghe(nhấn giữ yên 1 lúc chờ đèn tắt là xong), un-pair tất cả 2 loa từ phone nếu như đã lỡ pair trước đó và tắt bluetooth của phone. Bước 2: gắn tai nghe LEFT vào tai rồi chạm vào cảm ứng và giữ yên đến khi nghe chữ connecting thì buông ra rồi chờ tầm 5-10s rồi tới bước 3. Bước 3: gắn tai nghe RIGHT vào tai rồi chạm vào cảm ứng và giữ yên đến khi nghe chữ connecting thì buông ra rồi chờ 1 chút sẽ nghe 2 phone nói cùng 1 lúc chữ connecting. Bước 4: Mở bluetooth trên phone và search, phải chỉ hiện ra 1 tên thôi là đúng. Bước 5: nếu trong phone thấy 2 tên thì tắt bluetooth phone, tắt cả 2 tai nghe và làm lại từ đầu!======= -1* vì shop ko hướng dẫn kỹ cách pair nên tốn thời gian quá =========== ## Aspect: shop
output: [aooe] ko hướng dẫn kỹ cách pair
Ví dụ 5:
input: làm sao để kết nối chế độ kép, mún mỗi cái kết nối 1 cái điện thoại, mà khi bỏ ra khỏi dock sạc thì 2 tai đã tự kết nối với nhau ## Aspect: kết nối chế độ kép,
output: [aooe] làm sao để
Ví dụ 6:
input: Dùng được trong tầm giá. Nếu có điều kiện thì mua quạt xịn hơn chạy cho êm. ## Aspect: tầm giá.
output: [aooe] Dùng được
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ############################## AOPE ##############################
        self.aope["example"] = """Ví dụ 1:
input: hay vlinnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
output: [aope] <IA> $ hay vlinnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
Ví dụ 2:
input: Hài lòng với sản phẩm nhận được. Mong rằng tiki sẽ luôn có nhiều sản phẩm tốt.
output: [aope] sản phẩm $ Hài lòng ## sản phẩm $ tốt.
Ví dụ 3:
input: mới mua được 2 tuần. giờ ngôn ngữ chuyển sang trung quốc. pin thì được 15p đã hết
output: [aope] ngôn ngữ $ chuyển sang trung quốc. ## pin $ được 15p đã hết
Ví dụ 4:
input: Bé nhà mình bé trai mà trong đơn hàng không cho ghi chú. Mình sợ giao ngẫu nhiên lại nhầm màu hồng bé gái
output: [aope] đơn hàng $ không cho ghi chú. ## giao ngẫu nhiên $ sợ giao ngẫu nhiên lại nhầm màu hồng
Ví dụ 5:
input: chấp nhận được, hông quá tệ cũng hông quá xấu, trung bình.
output: [aope] <IA> $ chấp nhận được, hông quá tệ ## <IA> $ quá xấu, trung bình.
Ví dụ 6:
input: Mình mua 2 cái, phí cồng kềnh 26k, ahihi Hàng như hình
output: [aope] phí cồng kềnh $ 26k, ## Hàng $ như hình
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]

        ############################## AOSTE ############################## 
        self.aoste["example"] = """Ví dụ 1:
input: Giao hàng nhanh, chất lượng quạt chạy ổn, phù hợp giá tiền.
output: [aoste] Giao hàng $ nhanh, $ POS ## quạt chạy $ ổn, $ POS ## giá tiền. $ phù hợp $ POS
Ví dụ 2:
input: Sản phẩm ok, giao hàng nhanh, dịch vụ cũng tốt, sẽ mua hàng nữa nếu cần
output: [aoste] Sản phẩm $ ok, $ POS ## giao hàng $ nhanh, $ POS ## dịch vụ $ cũng tốt, $ POS
Ví dụ 3:
input: Rất ko hài lòng. Giao hàng lỗi ko lắp được nắp đậy
output: [aoste] <IA> $ ko hài lòng. $ NEG ## Giao hàng $ lỗi $ NEG ## <IA> $ ko lắp được nắp đậy $ NEG
Ví dụ 4:
input: Chưa nhận được đơn hàng. Hôm đó tôi nghỉ phép, không biết giao cho ai.
output: [aoste] đơn hàng. $ Chưa nhận được $ NEG
Ví dụ 5:
input: Mình muốn mua trả góp nhưng không biết phải thanh toán như thế nào? Tư vấn giúp mình!
output: [aoste] Tư vấn $ no_opinion $ NEU
Ví dụ 6:
input: sản phẩm đây là mặt trống hay nguyên cái trống zậy shop
output: [aoste] sản phẩm $ no_opinion $ NEU ## mặt trống $ no_opinion $ NEU ## nguyên cái trống $ no_opinion $ NEU
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]

    @override
    def load_instruction_2_modified(self):
        """ Instruction với 1 positive example, 1 negative example và 1 neutral example """
        
        ############################## ATE ##############################
        self.ate["example"] = """Ví dụ 1:
input: sản phẩm giao khá nhanh và đóng gói cẩn thận nội dung khá gay cấn
output: [ate] đóng gói ## giao ## nội dung
Ví dụ 2:
input: Ấm ko chắc chắn lắm, bị cọ sát xước sơn bên ngoài, miệng ấm hơi rộng, nếu rót nước gần hết thì nắp ấm há mồm chảy nước ra theo.
output: [ate] Ấm ## miệng ấm ## nắp ấm ## sơn
Ví dụ 3:
input: đang sử dụng nên chưa biết là có tốt hay không
output: [ate] sử dụng
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ############################## ATSC ##############################
        self.atsc["example"] = """Ví dụ 1:
input: Kéo béng thật, cầm chắc tay, tôi đã mua thử 1 cái, ok quá đã mua thêm 1 cái ## Aspect: cầm
output: [atsc] POS
Ví dụ 2:
input: Phiền hà khách hàng khi phải đủ 25 kí tự ms gửi dc ## Aspect: gửi dc
output: [atsc] NEG
Ví dụ 3:
input: Thật khó có thể trách tiki vì đợt này giao hàng muộn vì nhiều lý do, lý do nhà in chậm, vận chuyển, hệ thống lớn, tuy bản thân phải chờ khá sốt ruột rồi nhưng chuyện gì đã qua thì nên cho qua, chỉ tự trách mình là người nhận sau :3 ## Aspect: giao hàng
output: [atsc] NEU
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]

        ############################## ASPE ##############################
        self.aspe["example"] = """Ví dụ 1:
input: hàng như quảng cáo.nhìn chung là dễ thương
output: [aspe] hàng $ POS ## <IA> $ POS
Ví dụ 2:
input: Gối ôm quá ngắn, cứng quá,.............................
output: [aspe] Gối ôm $ NEG ## <IA> $ NEG
Ví dụ 3:
input: uống cũng tạm dc viết cho đủ 50 từ mà có vẻ dài quá
output: [aspe] uống $ NEU
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]

        ############################## AOOE ##############################
        self.aooe["example"] = """Ví dụ 1:
input: Cực kì hài lòng từ giao hàng đến chất lượng sản phẩm . ## Aspect: chất lượng sản phẩm
output: [aooe] Cực kì hài lòng
Ví dụ 2:
input: balo có 2 nút bấm để khoá lại một bên bị rớt ra mua về phải đi đóng lại . bán hàng ko kiểm tra sp trước khi giao cho kh ## Aspect: bán hàng
output: [aooe] ko kiểm tra sp
Ví dụ 3:
input: làm sao để kết nối chế độ kép, mún mỗi cái kết nối 1 cái điện thoại, mà khi bỏ ra khỏi dock sạc thì 2 tai đã tự kết nối với nhau ## Aspect: kết nối chế độ kép,
output: [aooe] làm sao để
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ############################## AOPE ##############################
        self.aope["example"] = """Ví dụ 1:
input: hay vlinnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
output: [aope] <IA> $ hay vlinnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
Ví dụ 2:
input: mới mua được 2 tuần. giờ ngôn ngữ chuyển sang trung quốc. pin thì được 15p đã hết
output: [aope] ngôn ngữ $ chuyển sang trung quốc. ## pin $ được 15p đã hết
Ví dụ 3:
input: chấp nhận được, hông quá tệ cũng hông quá xấu, trung bình.
output: [aope] <IA> $ chấp nhận được, hông quá tệ ## <IA> $ quá xấu, trung bình.
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]

        ############################## AOSTE ############################## 
        self.aoste["example"] = """Ví dụ 1:
input: Giao hàng nhanh, chất lượng quạt chạy ổn, phù hợp giá tiền.
output: [aoste] Giao hàng $ nhanh, $ POS ## quạt chạy $ ổn, $ POS ## giá tiền. $ phù hợp $ POS
Ví dụ 2:
input: Rất ko hài lòng. Giao hàng lỗi ko lắp được nắp đậy
output: [aoste] <IA> $ ko hài lòng. $ NEG ## Giao hàng $ lỗi $ NEG ## <IA> $ ko lắp được nắp đậy $ NEG
Ví dụ 3:
input: Mình muốn mua trả góp nhưng không biết phải thanh toán như thế nào? Tư vấn giúp mình!
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]

    def load_instruction_3(self):
        """2 example với đầy đủ POS NEU và NEG"""
        
        ############################## ATE ##############################
        self.ate["example"] = """Ví dụ 1:
input: Áo bị rách, chất vải mỏng nhưng với giá tiền thì phù hợp, size rộng, tôi nghĩ không cho giây vào sẽ đẹp hơn ( giây rất xấu không khác gì giây quần )    :)))))))))
output: [ate] giây ## giá tiền ## chất vải ## size ## Áo
Ví dụ 2:
input: chất lượng tạm. chỉ format FAT32 thì copy nhanh, chuẩn NTFS thì chậm rì
output: [ate] copy ## chất lượng ## chuẩn NTFS
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ############################## ATSC ##############################
        self.atsc["example"] = """Ví dụ 1:
input: Tiki gói bọc cẩn thận nhưng sản phẩm thì bị gỉ. Với mức giá 22.000 đồng thì tạm chấp nhận. Mọi người mua kích cỡ này sẽ rất phù hợp trồng những cây để bàn như sen đá. ## Aspect: giá
output: [atsc] NEU
Ví dụ 2:
input: Tiki nên kiểm tra lại hàng hoá và cách đóng gói. Hàng nhận về bị bể mất một mảng lớn ở nắp như vậy. Chả biết hộp bị sẵn hay do vận chuyển. May là lần này không bị ảnh hưởng đến sản phẩm bên trong, chứ còn một số hàng hoá dễ vỡ khác thì như nào nữa. ## Aspect: Tiki
output: [atsc] NEU
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]

        ############################## ASPE ##############################
        self.aspe["example"] = """Ví dụ 1:
input: Tôi Uống gần hết 2 hộp ( ủ 3 gói vào khoảng 1,5l nước uống cả ngày )  cân nặng xuống nhẹ , nhưng 2 ngày nay mụn nổi khá nhiều ở người và mặt Không biết là nên sử dụng tiếp hay ngưng lại vậy ạ
output: [aspe] Uống $ POS ## <IA> $ NEG ## sử dụng $ NEU
Ví dụ 2:
input: Điện thoại bị trầy dán vào thấy đỡ xấu hơn hẳn :)) . Tiện dụng nhưng hơi nhỏ, bỏ được tiền không nhiều, nhưng tầm giá vậy thì không đòi hỏi thêm
output: [aspe] dán vào $ POS ## <IA> $ POS ## <IA> $ NEG ## bỏ được tiền $ NEG ## tầm giá $ NEU
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]

        ############################## AOOE ##############################
        self.aooe["example"] = """Ví dụ 1:
input: mua cho thằng em ở nhà đọc, bao bì cẩn thận nhưng chắc do khâu vận chuyển nên các góc của bìa sách bị cong, cỡ chữ trung bình, chất lượng giấy ổn ## Aspect: bao bì
output: [aooe] cẩn thận
Ví dụ 2:[ate] vải ## Số 1 ## co dãn,[ate] co dãn ## IE ## Số 1 ## vải,
input: Sản phẩm tốt, có điều hàng quốc tế nên giao hàng hơi chậm ## Aspect: Sản phẩm
output: [aooe] tốt,
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ############################## AOPE ##############################
        self.aope["example"] = """Ví dụ 1:
input: san pham tam duoc, lúc nhận hàng bị nứt 1 tí nhưng ko sao
output: [aope] san pham $ tam duoc, ## <IA> $ bị nứt 1 tí ## <IA> $ ko sao
Ví dụ 2:
input: Về chất lượng sản phẩm: vải mềm, tốt, mặc mát mẻ, thoải mái. Về giao hàng: shop nói giao hàng màu ngẫu nhiên nhưng giao 2/3 cái là màu đỏ. Như vậy không hay cho lắm, nếu giao 3 màu khác nhau thì tốt hơn. Hy vọng sẽ cải tiến quy trình bán hàng.
output: [aope] vải $ mềm, tốt, ## mặc $ mát mẻ, thoải mái. ## giao hàng màu $ no_opinion ## <IA> $ không hay cho
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]

        ############################## AOSTE ############################## 
        self.aoste["example"] = """Ví dụ 1:
input: Sạc có 2 dây cho Samsung và iPhone. Z mà khi mình nhận được thì chỉ sạc cho samsung còn iPhone k vào. But đổi lại mình thích dịch vụ của Tiki. Mình đã gửi hàng đổi lại và đang chờ Tiki ktra
output: [aoste] Sạc $ có 2 dây cho Samsung và iPhone. $ NEU ## iPhone $ k vào. $ NEG ## dịch vụ của Tiki. $ thích $ POS ## gửi hàng đổi lại $ no_opinion $ NEU ## Tiki $ no_opinion $ NEU
Ví dụ 2:
input: sản phẩm có gai mềm, rửa mặt nhanh và sạch,  phần đế hút ko giữ đc khi dán vào mặt thẳng đứng, tuy nhiên có thể dính vào mặt bàn nằm ngang thì OK rồi. nhựa hơi mùi nhẹ thôi, ko phải vấn đề. Rất tiện lợi
output: [aoste] gai $ mềm, $ POS ## rửa mặt $ nhanh $ POS ## rửa mặt $ sạch, $ POS ## phần đế hút $ ko giữ đc $ NEG ## dính vào mặt bàn nằm ngang $ OK $ POS ## nhựa $ hơi mùi nhẹ $ NEU ## <IA> $ Rất tiện lợi $ POS
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]
    
    

class InstructionSpanHandler(BaseInstructionHandler):
		
    @override
    def load_instruction_1(self):
        ############################### ATE ###############################
        self.ate["example"] = """Ví dụ 1:
input: chất lượng ok.
output: [ate] chất lượng
Ví dụ 2:
input: Đó giờ chỉ tin dùng hạt chia của shop realfood. Hài lòng và luôn mua lại của shop này.
output: [ate] shop realfood. ## shop này.
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ##################### atsc #####################
        self.atsc["example"] = """Ví dụ 1:
input: SP tốt rất ưng ý ## Aspect: SP
output: [atsc] POS
Ví dụ 2:
input: nước sơn rất lì. ## Aspect: nước sơn
output: [atsc] POS
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]
        ##################### aspe #####################
        self.aspe["example"] = """Ví dụ 1:
input: đóng mở dễ dàng
output: [aspe] đóng mở $ POS
Ví dụ 2:
input: Mua đợt sale giá ngon,
output: [aspe] giá $ POS
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]
        ##################### aooe #####################
        self.aooe["example"] = """Ví dụ 1:
input: dùng ok ## Aspect: dùng
output: [aooe] ok
Ví dụ 2:
input: Giấy cứng ,chất lượng. ## Aspect: Giấy
output: [aooe] cứng ,chất lượng.
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ##################### aope #####################
        self.aope["example"] = """Ví dụ 1:
input: rất chắc chắn.
output: [aope] <IA> $ rất chắc chắn.
Ví dụ 2:
input: Tốt
output: [aope] <IA> $ Tốt
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]
        ##################### aoste #####################
        self.aoste["example"]= """Ví dụ 1:
input: đóng gói cẩn thận.
output: [aoste] đóng gói $ cẩn thận. $ POS
Ví dụ 2:
input: dễ lắp ráp
output: [aoste] lắp ráp $ dễ $ POS
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]

    @override
    def load_instruction_2(self):
        ############################### ATE ###############################
        self.ate["example"] = """Ví dụ 1:
input: chất lượng ok.
output: [ate] chất lượng
Ví dụ 2:
input: Đó giờ chỉ tin dùng hạt chia của shop realfood. Hài lòng và luôn mua lại của shop này.
output: [ate] shop realfood. ## shop này.
Ví dụ 3:
input: sau khi thoa thấm ko nhanh,
output: [ate] thoa
Ví dụ 4:
input: may ẩu bị bun chỉ ù chưa xử dụng
output: [ate] may ## chỉ
Ví dụ 5:
input: BCS mỏng quá sức ak.
output: [ate] BCS
Ví dụ 6:
input: Tạm chấp nhận trong tầm giá
output: [ate] giá
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ##################### atsc #####################
        self.atsc["example"] = """Ví dụ 1:
input: SP tốt rất ưng ý ## Aspect: SP
output: [atsc] POS
Ví dụ 2:
input: nước sơn rất lì. ## Aspect: nước sơn
output: [atsc] POS
Ví dụ 3:
input: ghi 2m giao 1m ## Aspect: giao
output: [atsc] NEG
Ví dụ 4:
input: màn hình màu k được tươi tắn ## Aspect: màn hình màu
output: [atsc] NEG
Ví dụ 5:
input: Mới nhận hàng ## Aspect: nhận hàng
output: [atsc] NEU
Ví dụ 6:
input: mới sd nên chưa biết coing dụng ra sao ## Aspect: coing dụng
output: [atsc] NEU
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]
        ##################### aspe #####################
        self.aspe["example"] = """Ví dụ 1:
input: đóng mở dễ dàng
output: [aspe] đóng mở $ POS
Ví dụ 2:
input: Mua đợt sale giá ngon,
output: [aspe] giá $ POS
Ví dụ 3:
input: Hàng kém chất lượng.
output: [aspe] Hàng $ NEG
Ví dụ 4:
input: dây sạc thì dơ
output: [aspe] dây sạc $ NEG
Ví dụ 5:
input: Về chất lượng chưa rõ
output: [aspe] chất lượng $ NEU
Ví dụ 6:
input: sản phẩm béo.
output: [aspe] sản phẩm $ NEU
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]
        ##################### aooe #####################
        self.aooe["example"] = """Ví dụ 1:
input: dùng ok ## Aspect: dùng
output: [aooe] ok
Ví dụ 2:
input: Giấy cứng ,chất lượng. ## Aspect: Giấy
output: [aooe] cứng ,chất lượng.
Ví dụ 3:
input: gắn lại không chặt được ## Aspect: gắn lại
output: [aooe] không chặt được
Ví dụ 4:
input: ko thích hợp nghe nhạc nhẹ ## Aspect: nghe nhạc nhẹ
output: [aooe] ko thích hợp
Ví dụ 5:
input: Không biết phải đổi hàng ntn? ## Aspect: đổi hàng
output: [aooe] <IO>
Ví dụ 6:
input: Dạng tản văn, ## Aspect: Dạng tản văn,
output: [aooe] <IO>
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ##################### aope #####################
        self.aope["example"] = """Ví dụ 1:
input: rất chắc chắn.
output: [aope] <IA> $ rất chắc chắn.
Ví dụ 2:
input: Tốt
output: [aope] <IA> $ Tốt
Ví dụ 3:
input: giá khá cao
output: [aope] giá $ khá cao
Ví dụ 4:
input: 2 cái 2 xương rồng mình đặt k có chữ
output: [aope] 2 cái 2 xương rồng $ k có chữ
Ví dụ 5:
input: chưa sd nên không biết có bền không.
output: [aope] bền $ không biết
Ví dụ 6:
input: chất lượng thì dùng lâu mới đánh giá đươc
output: [aope] chất lượng $ dùng lâu mới đánh giá đươc
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]
        ##################### aoste #####################
        self.aoste["example"]= """Ví dụ 1:
input: đóng gói cẩn thận.
output: [aoste] đóng gói $ cẩn thận. $ POS
Ví dụ 2:
input: dễ lắp ráp
output: [aoste] lắp ráp $ dễ $ POS
Ví dụ 3:
input: lắp ống vào vòi không kĩ làm rỉ nước.
output: [aoste] lắp ống vào vòi $ không kĩ làm rỉ nước. $ NEG
Ví dụ 4:
input: hơi khó sử dụng.
output: [aoste] sử dụng. $ hơi khó $ NEG
Ví dụ 5:
input: Chưa biết tác dụng có hiệu quả hay không
output: [aoste] tác dụng $ <IO> $ NEU
Ví dụ 6:
input: đóng gói
output: [aoste] đóng gói $ <IO> $ NEU
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]


    @override
    def load_instruction_2_modified(self):
        ############################### ATE ###############################
        self.ate["example"] = """Ví dụ 1:
input: Giòn,
output: [ate] <IA>
Ví dụ 2:
input: giao hàng kiểu gì mà gối ướt hết.
output: [ate] gối ## giao hàng
Ví dụ 3:
input: gọi hãng bảo hành.
output: [ate] bảo hành.
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ##################### atsc #####################
        self.atsc["example"] = """Ví dụ 1:
input: size chuẩn. ## Aspect: size
output: [atsc] POS
Ví dụ 2:
input: Thiết kế khay chưa hợp lý, không để được nhiều bát. ## Aspect: Thiết kế khay
output: [atsc] NEG
Ví dụ 3:
input: Mong Tiki nhanh chóng tìm ra giải pháp cho vấn đề này. ## Aspect: Tiki
output: [atsc] NEU
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]
        ##################### aspe #####################
        self.aspe["example"] = """Ví dụ 1:
input: Áo có 2 mặt sử dụng,
output: [aspe] Áo $ POS
Ví dụ 2:
input: vận chuyển có hơi lâu thôi,
output: [aspe] vận chuyển $ NEG
Ví dụ 3:
input: mình chưa mặc thử. nên k biết hợp ko.
output: [aspe] mặc $ NEU
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]
        ##################### aooe #####################
        self.aooe["example"] = """
        Ví dụ 1:
input: Màu có thể dùng cho nhiều dịp. ## Aspect: Màu
output: [aooe] có thể dùng cho nhiều dịp.
Ví dụ 2:
input: Sản phẩm lúc giao bị cắt bịch, ## Aspect: bịch,
output: [aooe] bị cắt
Ví dụ 3:
input: Hàng mua chủ yếu làm phụ kiện đồ chơi cho trẻ con, ## Aspect: Hàng
output: [aooe] làm phụ kiện đồ chơi cho trẻ con,
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ##################### aope #####################
        self.aope["example"] = """Ví dụ 1:
input: Quyển sách rất hấp dấn, thu hút người đọc
output: [aope] Quyển sách $ hấp dấn, thu hút người đọc
Ví dụ 2:
input: độ bền quá kém,
output: [aope] độ bền $ quá kém,
Ví dụ 3:
input: Để xem dùng có đúng với số trang quảng cáo không.
output: [aope] dùng $ no_opinion
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]
        ##################### aoste #####################
        self.aoste["example"]= """Ví dụ 1:
input: hàng ngon
output: [aoste] hàng $ ngon $ POS
Ví dụ 2:
input: Máy hoạt động hơi lớn
output: [aoste] hoạt động $ hơi lớn $ NEG
Ví dụ 3:
input: Đã dùng dầu gội - xả đc 2 tuần,
output: [aoste] dùng $ no_opinion $ NEU
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]





class InstructionSegmentHandler(BaseInstructionHandler):

    @override
    def load_instruction_1(self):
        """ Instruction với 2 posivie example """
        
        ############################## ATE ##############################
        self.ate["example"] = """Ví dụ 1:
input: ăn rất ngon , shop rất uy_tín , mình sẽ quay lại khi dùng hết
output: [ate] ăn ## shop
Ví dụ 2:
input: Dán dính rat tốt , giá lại rất rẻ . Nói_chung là mua ko hối_tiếc
output: [ate] <IA> ## giá ## Dán dính
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ############################## ATSC ##############################
        self.atsc["example"] = """Ví dụ 1:
input: Giao hàng nhanh , an_toàn cho sp Shop nhiệt_tình Sp thì khỏi bàn . Like ## Aspect: Sp
output: [atsc] POS
Ví dụ 2:
input: Pin đẹp , pin siêu trâu , sản_phẩm đúng với mô_tả . Ủng_hộ shop . ## Aspect: Pin
output: [atsc] POS
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]

        ############################## ASPE ##############################
        self.aspe["example"] = """Ví dụ 1:
input: Nên mua . Nên mua . Nên mua . Nên mua . Nên mua . Nên mua .
output: [aspe] <IA> $ POS ## <IA> $ POS ## <IA> $ POS ## <IA> $ POS ## <IA> $ POS ## <IA> $ POS
Ví dụ 2:
input: đóng_gói cẩn_thận mới_đầu năm mà giao hàng nhanh rất vui
output: [aspe] đóng_gói $ POS ## giao hàng $ POS ## <IA> $ POS
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]

        ############################## AOOE ##############################
        self.aooe["example"] = """Ví dụ 1:
input: Hàng rất dễ_thương , dùng rất tiện ! Sẽ tiếp_tục ủng_hộ tiki ## Aspect: dùng
output: [aooe] rất tiện
Ví dụ 2:
input: Rất ok đèn sáng đều chuyển màu đẹp_mắt , rất hài_lòng ## Aspect: <IA>
output: [aooe] ok ## hài_lòng
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ############################## AOPE ##############################
        self.aope["example"] = """Ví dụ 1:
input: rất hay để cho con đọc và tốt hơn nhiều điện_thoại
output: [aope] <IA> $ rất hay ## <IA> $ tốt hơn
Ví dụ 2:
input: San pham tot , ok , ..................................................................................................
output: [aope] San pham $ tot ## <IA> $ ok
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]

        ############################## AOSTE ############################## 
        self.aoste["example"] = """Ví dụ 1:
input: Nhỏ gọn , hoàn_thiện tốt , lỗ cắm chắc_chắn , sạc ổn đinh !
output: [aoste] <IA> $ Nhỏ gọn $ POS ## hoàn_thiện $ tốt $ POS ## lỗ cắm $ chắc_chắn $ POS ## sạc $ ổn đinh $ POS
Ví dụ 2:
input: tiền nào của đó .. ưng_ý . rất ok . vãi đẹp đường may ok ...
output: [aoste] <IA> $ tiền nào của đó $ POS ## <IA> $ ưng_ý $ POS ## <IA> $ rất ok $ POS ## vãi $ đẹp $ POS ## đường may $ ok $ POS
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]

    @override		
    def load_instruction_2(self):
        """ Instruction với 2 positive example, 2 negative example và 2 neutral example """
        
        ############################## ATE ##############################
        self.ate["example"] = """Ví dụ 1:
input: ăn rất ngon , shop rất uy_tín , mình sẽ quay lại khi dùng hết
output: [ate] ăn ## shop
Ví dụ 2:
input: Dán dính rat tốt , giá lại rất rẻ . Nói_chung là mua ko hối_tiếc
output: [ate] <IA> ## giá ## Dán dính
Ví dụ 3:
input: phải dùng một thời_gian mới có đánh_giá được độ bền ... tiền_nào_của_nấy
output: [ate] <IA>
Ví dụ 4:
input: Dùng được trong tầm giá . Nếu có điều_kiện thì mua quạt xịn hơn chạy cho êm .
output: [ate] tầm giá
Ví dụ 5:
input: thiếu một dây kết_nối 3.5 mm đè nghị shop gửi bổ_sung và chịu ship
output: [ate] dây kết_nối 3.5
Ví dụ 6:
input: Không như kỳ_vọng . Mỏng và mềm oặc ẹo , không có đứng . Nói_chung mua về còn vứt đó chưa biết dùng để làm_gì
output: [ate] <IA> ## dùng
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ############################## ATSC ##############################
        self.atsc["example"] = """Ví dụ 1:
input: Giao hàng nhanh , an_toàn cho sp Shop nhiệt_tình Sp thì khỏi bàn . Like ## Aspect: Sp
output: [atsc] POS
Ví dụ 2:
input: Pin đẹp , pin siêu trâu , sản_phẩm đúng với mô_tả . Ủng_hộ shop . ## Aspect: Pin
output: [atsc] POS
Ví dụ 3:
input: Mình vừa đặt mua bao ipad ari2 nhưng phiên_bản ari2 mình 2017 độ dầy 75mm mà bản cũ 61 mm liệu có lắp vừa không ai ## Aspect: lắp vừa
output: [atsc] NEU
Ví dụ 4:
input: tôi vừa mua với giá 15.449.000 , mới nhận hàng sáng nay . giờ lên xem giá còn 15.199.000 . giờ đổi trả hàng xong mua lại để được giá thấp hơn được không ? 🙄🙄🙄 ## Aspect: giá
output: [atsc] NEU ## NEU
Ví dụ 5:
input: Thất_vọng , trong đây ghi có dành cho ipad mini 123 mà rốt_cuộc giao hàng lại là ipad 234 gửi trả lại . Tiếp_tục nhận lần 2 vẫn lại là loại này . Nên quyết_định trả lại luôn không mua nữa . Mà thấy màu và chất_lượng không tốt cũng không đẹp ## Aspect: chất_lượng
output: [atsc] NEG ## NEG
Ví dụ 6:
input: Kiểu mang 1 lần là vứt , không sử_dụng lại đc nưa . vớ rất dễ giãn ## Aspect: sử_dụng
output: [atsc] NEG
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]

        ############################## ASPE ##############################
        self.aspe["example"] = """Ví dụ 1:
input: Nên mua . Nên mua . Nên mua . Nên mua . Nên mua . Nên mua .
output: [aspe] <IA> $ POS ## <IA> $ POS ## <IA> $ POS ## <IA> $ POS ## <IA> $ POS ## <IA> $ POS
Ví dụ 2:
input: đóng_gói cẩn_thận mới_đầu năm mà giao hàng nhanh rất vui
output: [aspe] đóng_gói $ POS ## giao hàng $ POS ## <IA> $ POS
Ví dụ 3:
input: chưa dùng nên chưa biết . mua 3 gói dk tặng 1 xe_đạp cho bé
output: [aspe] dùng $ NEU ## mua $ NEU
Ví dụ 4:
input: chất vải cũng bình_thường thôi so với giá tiền như_vậy .
output: [aspe] chất vải $ NEU ## giá tiền $ NEU
Ví dụ 5:
input: Mình dùng rồi kiến ít bâu vào Và không có hết kiến Thuốc nhanh bị chảy nước
output: [aspe] dùng $ NEG ## dùng $ NEG ## Thuốc $ NEG
Ví dụ 6:
input: Tg là sữa có đường nhưg đây lại là sữa không đường mua về chả ai uống
output: [aspe] sữa $ NEG ## uống $ NEG
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]

        ############################## AOOE ##############################
        self.aooe["example"] = """Ví dụ 1:
input: Hàng rất dễ_thương , dùng rất tiện ! Sẽ tiếp_tục ủng_hộ tiki ## Aspect: dùng
output: [aooe] rất tiện
Ví dụ 2:
input: Rất ok đèn sáng đều chuyển màu đẹp_mắt , rất hài_lòng ## Aspect: <IA>
output: [aooe] ok ## hài_lòng
Ví dụ 3:
input: Hiện_nay còn chương_trình mua 2 enfa 4 tặng xe ko bạn ơi ## Aspect: chương_trình mua 2 enfa 4 tặng xe
output: [aooe] no_opinion
Ví dụ 4:
input: chấp_nhận được , hông quá tệ cũng hông quá xấu , trung_bình . ## Aspect: <IA>
output: [aooe] chấp_nhận được , hông quá tệ ## quá xấu , trung_bình
Ví dụ 5:
input: Khâu lấy cafe sau khi xay hơi bất_tiện , khó_khăn . Vệ_sinh cũng khó . ## Aspect: Vệ_sinh
output: [aooe] cũng khó
Ví dụ 6:
input: Mùi nhựa khá là khét . Đế làm khá èo_uột . Vài hôm bẹp dí ## Aspect: <IA>
output: [aooe] bẹp dí
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ############################## AOPE ##############################
        self.aope["example"] = """Ví dụ 1:
input: rất hay để cho con đọc và tốt hơn nhiều điện_thoại
output: [aope] <IA> $ rất hay ## <IA> $ tốt hơn
Ví dụ 2:
input: San pham tot , ok , ..................................................................................................
output: [aope] San pham $ tot ## <IA> $ ok
Ví dụ 3:
input: làm_sao để kết_nối chế_độ kép , mún mỗi cái kết_nối 1 cái điện_thoại , mà khi bỏ ra khỏi dock sạc thì 2 tai đã tự kết_nối với nhau
output: [aope] kết_nối chế_độ kép $ làm_sao để ## 2 tai $ tự kết_nối với nhau
Ví dụ 4:
input: Tỷ_lệ chuyển_đổi là do đặc_tính của cell pin mà shop , tuỳ theo nhà sa3b xuất mà pin có tỷ_lệ chuyển_đổi khác nhau
output: [aope] Tỷ_lệ chuyển_đổi $ no_opinion ## tỷ_lệ chuyển_đổi $ khác nhau
Ví dụ 5:
input: Áo quá xấu lại mỏng nữa , ngắn giao hình vớ_vẩn không giống các hình trên mô_tả sản_phẩm , xấu 1 cách thậm_tệ . Quá chán
output: [aope] Áo $ quá xấu ## Áo $ mỏng ## Áo $ ngắn ## giao hình $ vớ_vẩn ## sản_phẩm $ không giống các hình trên mô_tả ## <IA> $ xấu 1 cách thậm_tệ ## <IA> $ Quá chán
Ví dụ 6:
input: hạt nhỏ , trà bình_thường , vị nhẹ , ko đủ tiêu_chuẩn búp 3 lá
output: [aope] hạt $ nhỏ ## trà $ bình_thường ## vị $ nhẹ ## tiêu_chuẩn búp 3 lá $ ko đủ
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]

        ############################## AOSTE ############################## 
        self.aoste["example"] = """Ví dụ 1:
input: Nhỏ gọn , hoàn_thiện tốt , lỗ cắm chắc_chắn , sạc ổn đinh !
output: [aoste] <IA> $ Nhỏ gọn $ POS ## hoàn_thiện $ tốt $ POS ## lỗ cắm $ chắc_chắn $ POS ## sạc $ ổn đinh $ POS
Ví dụ 2:
input: tiền nào của đó .. ưng_ý . rất ok . vãi đẹp đường may ok ...
output: [aoste] <IA> $ tiền nào của đó $ POS ## <IA> $ ưng_ý $ POS ## <IA> $ rất ok $ POS ## vãi $ đẹp $ POS ## đường may $ ok $ POS
Ví dụ 3:
input: đang sử dụng nên chưa biết là có tốt hay không
output: [aoste] sử dụng $ chưa biết là có tốt hay không $ NEU
Ví dụ 4:
input: Nước_cốt gà dành cho những người hoạt_động trí_óc như mình hoặc làm_việc với cường_độ cao
output: [aoste] Nước_cốt gà $ dành cho những người hoạt_động trí_óc $ NEU ## Nước_cốt gà $ làm_việc với cường_độ cao $ NEU
Ví dụ 5:
input: sp ko phát đc âm_thanh . dù đả kết_nối . thất_vọng về sp
output: [aoste] âm_thanh $ ko phát đc $ NEG ## sp $ thất_vọng $ NEG
Ví dụ 6:
input: Chất_lượng ko như mong_đợi ,_son tẩy tế_bào chết dễ gãy
output: [aoste] Chất_lượng $ ko như mong_đợi $ NEG ## ,_son tẩy tế_bào chết $ ,_son $ NEG ## ,_son tẩy tế_bào chết $ dễ gãy $ NEG
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]

    @override
    def load_instruction_2_modified(self):
        """ Instruction với 1 positive example, 1 negative example và 1 neutral example """
        
        ############################## ATE ##############################
        self.ate["example"] = """Ví dụ 1:
input: Dán dính rat tốt , giá lại rất rẻ . Nói_chung là mua ko hối_tiếc
output: [ate] <IA> ## giá ## Dán dính
Ví dụ 2:
input: Dùng được trong tầm giá . Nếu có điều_kiện thì mua quạt xịn hơn chạy cho êm .
output: [ate] tầm giá
Ví dụ 3:
input: Không như kỳ_vọng . Mỏng và mềm oặc ẹo , không có đứng . Nói_chung mua về còn vứt đó chưa biết dùng để làm_gì
output: [ate] <IA> ## dùng
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ############################## ATSC ##############################
        self.atsc["example"] = """Ví dụ 1:
input: Pin đẹp , pin siêu trâu , sản_phẩm đúng với mô_tả . Ủng_hộ shop . ## Aspect: Pin
output: [atsc] POS
Ví dụ 2:
input: tôi vừa mua với giá 15.449.000 , mới nhận hàng sáng nay . giờ lên xem giá còn 15.199.000 . giờ đổi trả hàng xong mua lại để được giá thấp hơn được không ? 🙄🙄🙄 ## Aspect: giá
output: [atsc] NEU ## NEU
Ví dụ 3:
input: Kiểu mang 1 lần là vứt , không sử_dụng lại đc nưa . vớ rất dễ giãn ## Aspect: sử_dụng
output: [atsc] NEG
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]

        ############################## ASPE ##############################
        self.aspe["example"] = """Ví dụ 1:
input: đóng_gói cẩn_thận mới_đầu năm mà giao hàng nhanh rất vui
output: [aspe] đóng_gói $ POS ## giao hàng $ POS ## <IA> $ POS
Ví dụ 2:
input: chất vải cũng bình_thường thôi so với giá tiền như_vậy .
output: [aspe] chất vải $ NEU ## giá tiền $ NEU
Ví dụ 3:
input: Tg là sữa có đường nhưg đây lại là sữa không đường mua về chả ai uống
output: [aspe] sữa $ NEG ## uống $ NEG
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]

        ############################## AOOE ##############################
        self.aooe["example"] = """Ví dụ 1:
input: Hàng rất dễ_thương , dùng rất tiện ! Sẽ tiếp_tục ủng_hộ tiki ## Aspect: dùng
output: [aooe] rất tiện
Ví dụ 2:
input: chấp_nhận được , hông quá tệ cũng hông quá xấu , trung_bình . ## Aspect: <IA>
output: [aooe] chấp_nhận được , hông quá tệ ## quá xấu , trung_bình
Ví dụ 3:
input: Khâu lấy cafe sau khi xay hơi bất_tiện , khó_khăn . Vệ_sinh cũng khó . ## Aspect: Vệ_sinh
output: [aooe] cũng khó
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ############################## AOPE ##############################
        self.aope["example"] = """Ví dụ 1:
input: rất hay để cho con đọc và tốt hơn nhiều điện_thoại
output: [aope] <IA> $ rất hay ## <IA> $ tốt hơn
Ví dụ 2:
input: làm_sao để kết_nối chế_độ kép , mún mỗi cái kết_nối 1 cái điện_thoại , mà khi bỏ ra khỏi dock sạc thì 2 tai đã tự kết_nối với nhau
output: [aope] kết_nối chế_độ kép $ làm_sao để ## 2 tai $ tự kết_nối với nhau
Ví dụ 3:
input: Áo quá xấu lại mỏng nữa , ngắn giao hình vớ_vẩn không giống các hình trên mô_tả sản_phẩm , xấu 1 cách thậm_tệ . Quá chán
output: [aope] Áo $ quá xấu ## Áo $ mỏng ## Áo $ ngắn ## giao hình $ vớ_vẩn ## sản_phẩm $ không giống các hình trên mô_tả ## <IA> $ xấu 1 cách thậm_tệ ## <IA> $ Quá chán
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]

        ############################## AOSTE ############################## 
        self.aoste["example"] = """Ví dụ 1:
input: Nhỏ gọn , hoàn_thiện tốt , lỗ cắm chắc_chắn , sạc ổn đinh !
output: [aoste] <IA> $ Nhỏ gọn $ POS ## hoàn_thiện $ tốt $ POS ## lỗ cắm $ chắc_chắn $ POS ## sạc $ ổn đinh $ POS
Ví dụ 2:
input: Nước_cốt gà dành cho những người hoạt_động trí_óc như mình hoặc làm_việc với cường_độ cao
output: [aoste] Nước_cốt gà $ dành cho những người hoạt_động trí_óc $ NEU ## Nước_cốt gà $ làm_việc với cường_độ cao $ NEU
Ví dụ 3:
input: sp ko phát đc âm_thanh . dù đả kết_nối . thất_vọng về sp
output: [aoste] âm_thanh $ ko phát đc $ NEG ## sp $ thất_vọng $ NEG
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]

    def load_instruction_3(self):
        """2 example với đầy đủ POS NEU và NEG"""
        
        ############################## ATE ##############################
        self.ate["example"] = """Ví dụ 1:
input: Điện_thoại bị trầy dán vào thấy đỡ xấu hơn hẳn : ) ) . Tiện_dụng nhưng hơi nhỏ , bỏ được tiền không nhiều , nhưng tầm giá vậy thì không đòi_hỏi thêm
output: [ate] <IA> ## tầm giá ## dán vào ## bỏ được tiền
Ví dụ 2:
input: đã mua sp trên tiki nhiều lần , hàng do tiki cung_cấp mình rất hài_lòng . lần này hàng k phải do tiki cung_cấp , sp rất tệ , nắp nồi bị hở , nên bay hết_hơi nước , nấu ít sẽ bị khô , cháy cháo . bảo_hành cho mình đi
output: [ate] <IA> ## hàng do tiki cung_cấp ## sp ## bảo_hành ## nấu ## nắp nồi
"""
        self.ate["instruction"] = self.ate["definition"] + self.ate["example"] + self.ate["delim_instruct"]

        ############################## ATSC ##############################
        self.atsc["example"] = """Ví dụ 1:
input: Nhận được serum vài ngày rồi nhưng hôm_nay mới dùng . Giao hàng hơi lâu nhưng được_cái bọc hàng rất cẩn_thận , cầm hộp mới nguyên lên không tí sứt_mẻ nào đã thích rồi . Mình mua của HAFA BEAUTY , date cũng mới từ tháng 6 thôi , nhận hàng một cái là cho luôn vào tủ_lạnh , đến hôm_nay mới mang ra dùng . Serum dạng lỏng , còn màu đỏ tươi chứng_tỏ chưa bị oxy_hoá , thoa lên da lúc đầu hơi có cảm_giác châm_chích nhưng chỉ tầm vài giây thôi . Serum lỏng nhưng apply lên thì thấm không nhanh lắm , cảm_giác hơi dính dính nhưng 5 phút là hết . Mới dùng lần đầu nên chưa thấy hiệu_quả gì mấy : 3 ## Aspect: <IA>
output: [atsc] POS ## NEG
Ví dụ 2:
input: Giao hàng nhanh . đặt 30/5 thì 31/5 đã giao nhưng dịch_vụ lắp_đặt thì đến 7/6 mới gọi . 4/6 bảo_hành điện_tử kích_hoạt . Tiki đóng_gói cẩn_thận mua Tiki rất yên_tâm . Tivi_LG 43 uj750t tuyệt_vời có magic remote theotivi xài rất đã nhưng Webos thì hơi chán tí . nếu lắp_đặt đúng như quảng_cáo 48 h thì quá tuyệt_vời . cám_ơn Tiki ## Aspect: mua Tiki
output: [atsc] POS
"""
        self.atsc["instruction"] = self.atsc["definition"] + self.atsc["example"] + self.atsc["delim_instruct"]

        ############################## ASPE ##############################
        self.aspe["example"] = """Ví dụ 1:
input: da mình khô đc khoảng 5 năm , từng thử sữa dưỡng thể của vaseline nhưng không hiệu_quả , dùng thử loại này hiệu_quả rõ_rệt trong vòng 1 tuần . Hiện h đã xài sd đc nửa bình rồi
output: [aspe] thử $ NEG ## dùng thử $ POS ## xài sd $ NEU
Ví dụ 2:
input: Tiki đóng cái hộp rõ to mở ra miếng dán bé chút ét , lột ra dán thử 1 cái cũng khá ok , giao hàng sớm hơn cả 1 tuần
output: [aspe] hộp $ NEU ## miếng dán $ NEG ## dán thử $ POS ## giao hàng $ POS
"""
        self.aspe["instruction"] = self.aspe["definition"] + self.aspe["example"] + self.aspe["delim_instruct"]

        ############################## AOOE ##############################
        self.aooe["example"] = """Ví dụ 1:
input: Sài khá ok và chất_lượng . kg xoá được vết ố nhưng tương_đối chập nhận dc ## Aspect: Sài
output: [aooe] khá ok ## chất_lượng
Ví dụ 2:
input: đèn chuyển về dùng ok . nhưng nên thêm chọn màu . mình hiểu là có mỗi màu đen , đến lúc nhận lại màu trắng . ## Aspect: nhận
output: [aooe] hiểu là có mỗi màu đen , đến lúc nhận lại màu trắng
"""
        self.aooe["instruction"] = self.aooe["definition"] + self.aooe["example"] + self.aooe["delim_instruct"]

        ############################## AOPE ##############################
        self.aope["example"] = """Ví dụ 1:
input: chất_lượng tạm . chỉ format FAT32 thì copy nhanh , chuẩn NTFS thì chậm_rì
output: [aope] chất_lượng $ tạm ## copy $ nhanh ## chuẩn NTFS $ chậm_rì
Ví dụ 2:
input: Hàng giống hình . Cao_su tốt . Mình hay đi size 39 nhưng đôi này mình đi hơi bó chân . Đi 1 thời_gian mới đánh_giá được chất_lượng . Giao hàng nhanh .
output: [aope] Hàng $ giống hình ## Cao_su $ tốt ## đi $ hơi bó chân ## chất_lượng $ mới đánh_giá được ## Giao hàng $ nhanh
"""
        self.aope["instruction"] = self.aope["definition"] + self.aope["example"] + self.aope["delim_instruct"]

        ############################## AOSTE ############################## 
        self.aoste["example"] = """Ví dụ 1:
input: Nhưng có_thể giao hang sớm hon duoc không ? Mua đơn hang dau tien giao rat nhanh .. don hang sau 4-5 ngay roi ma chua thay giao
output: [aoste] giao hang $ no_opinion $ NEU ## giao $ rat nhanh $ POS ## giao $ chua thay $ NEG
Ví dụ 2:
input: Áo bị rách , chất vải mỏng nhưng với giá tiền thì phù_hợp , size rộng , tôi nghĩ không cho giây vào sẽ đẹp hơn ( giây rất xấu không khác gì giây quần ) : ) ) ) ) ) ) ) ) )
output: [aoste] Áo $ bị rách $ NEG ## chất vải $ mỏng $ NEG ## giá tiền $ phù_hợp $ POS ## size $ rộng $ NEU ## giây $ rất xấu $ NEG
"""
        self.aoste["instruction"] = self.aoste["definition"] + self.aoste["example"] + self.aoste["delim_instruct"]